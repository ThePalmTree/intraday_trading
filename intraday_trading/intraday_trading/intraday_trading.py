import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV 
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from plotly.offline import plot
from plotly.graph_objs import Scattergl


##########################################################################################################
### DATA ###
##########################################################################################################


def dataset_building(n_max=None, verbose=None):
    ''' Build the dataset ''' 
    if verbose: print('Dataset building...')
    #cd=os.getcwd()
    stock_data=pd.read_csv('C:/Users/Loic/Documents/work/interview/maven/reference.data/stock.data.csv')
    index_data=pd.read_csv('C:/Users/Loic/Documents/work/interview/maven/reference.data/reference.data.csv')
       
    for data in [stock_data, index_data]:
        data.set_index(data.columns[0],inplace=True)
        data.index=pd.to_datetime(data.index,infer_datetime_format=True) # Speed optimized after testing
        # Make sure the date index is ascending, we avoid to sort because of the complexity
        data=data.sort_index(axis=0,ascending=True)
       
    # Cut the dataset to a lower number of obs
    if n_max is not None:
        stock_data=stock_data.iloc[-(n_max + 1):]       
        index_data=index_data.iloc[-(n_max + 1):]

    if verbose: print('Dataset built')

    return stock_data, index_data

def compute_prices(df):
    ''' Allow us to compute the history of prices given the input data
    Indeed the input  data provide returns since the last close
    We initialize all the prices at 100 at inception of our data '''
    res=pd.DataFrame(index=df.index, columns=df.columns)
    factor=pd.Series(100,index=df.columns) # we initialize the prices at 100
    last_timestamp=df.index[0]
    for i in df.index:
        if i.date() != last_timestamp.date(): # If we change the day
            factor=res.loc[last_timestamp,:] # then we take as factor the last closing price
            last_timestamp=i
        res.loc[i,:]=(1+df.loc[i,:])*factor # we multiply by the last closing price
    return res

def compute_returns(df, col_index=None, window=1):
    ''' Compute the returns for some preselected columns of the dataset
        And output them in a new dataset
    '''
    res=pd.DataFrame()
    if not col_index:
        col_index=df.columns
    for col in col_index:
        res[col] = df.loc[:,col]/df.loc[:,col].shift(window) - 1
    return res

stock_data,index_data=dataset_building(n_max=50000,verbose=1)
#data=stock_data.join(index_data,how='outer')
data=index_data
data.dropna(inplace=True)  
data_prices=compute_prices(data)
data_returns=compute_returns(data_prices)
data_returns.dropna(inplace=True)

##########################################################################################################
### MODEL ###
##########################################################################################################

# we build a model for each asset
lr={asset:LinearRegression() for asset in data_prices.columns}
lasso={asset:Lasso(normalize=True) for asset in data_prices.columns}
models=lasso
 
##########################################################################################################
### TRADING STRATEGY ###
##########################################################################################################

# We will work with a rolling window in the following way:
# Every 2 min we will look at the residuals of our assets versus our predicted assets
# the prediction will be based on the linear models built using the information from the rolling window


# Global params
rolling_window_size=720 # The number of data points we need to build our models, 720 for one day
transaction_cost=0.0 #0.0005 # in percent for a trade (entry and exit included), 5 bps by default
data_size=len(data_returns.index)
validation_set_size=int(data_size/4)
testing_set_size=data_size-validation_set_size


# Hyperparameters grid: 
hp_grid={'threshold':[1.5,2,3,4,5], # the limit on z score of the residual after which we trade
         'max_holding_period':[1,2,5,10,15,30,60]} # holding period of the positions


# We define a class to keep track of our trading strategy
class TradingStrategy():
    ''' This class will help us keep track of our trading strategy '''
    def __init__(self, models, data_returns, data_prices, rolling_window_size, validation_set_size, transaction_cost):
        self.models=models
        self.data_returns=data_returns
        self.data_prices=data_prices
        self.transaction_cost=transaction_cost
        self.rolling_window_size=rolling_window_size
        self.validation_set_size=validation_set_size
        self.ret=pd.Series(0.0,index=data_returns.index) # keep track of the realized returns
        self.m_resi=pd.Series(index=data_returns.index) # keep track of the max residuals
        self.m_asset=pd.Series(index=data_returns.index) # keep track of the asset on which we have the max resi
        self.pos=pd.Series(index=data_returns.columns)
        self.active_pos=False # this will help us keep track of our open position, and make sure we have only one strategy running at the same time
        self.holding_period=0
        self.entry_prices=pd.DataFrame()
        # Hyperparameters
        self.threshold=None
        self.max_holding_period=None

    def reinit(self):
        self.ret=pd.Series(0.0,index=data_returns.index) # keep track of the realized pnl
        self.active_pos=False # this will help us keep track of our open strategy, and make sure we have only one strategy running at the same time
        self.holding_period=0
        self.entry_prices=pd.DataFrame()
        
    def compute_resi(self, begin, end, verbose=False):
        ''' This function will compute the residuals on a certain period '''
        if verbose: print('Calculating Residuals...')
        for i in range(begin, end):
            today=self.data_returns.index[i]
            #if verbose: print(today) # debug
            train=self.data_returns.index[i-self.rolling_window_size:i-1]
            max_resi=0  
            max_asset=None
            for asset in self.data_returns.columns:  
                other_assets=self.data_returns.columns.drop(asset)
                Y=self.data_returns.loc[train,asset]
                X=self.data_returns.loc[train,other_assets]
                self.models[asset].fit(X, Y)
                resi=Y-self.models[asset].predict(X) # These are the residuals
                norm_resi=resi[-1]/np.std(resi) # This is our normalized last residual 
                if abs(norm_resi)>max_resi:
                    max_resi=norm_resi
                    max_asset=asset
            self.m_resi[today]=max_resi
            self.m_asset[today]=max_asset
        if verbose: print('Residuals calculated')        
     
    def compute_strat(self, begin, end, verbose=False):
        ''' This function will compute the strategy on a certain period, we need to compute the residuals first '''
        if verbose: print('Calculating Strategy...')
        for i in range(begin, end):     
            today=self.data_returns.index[i]   
            if not self.active_pos: # we look for a signal on our max residual
                 if abs(self.m_resi[today])>self.threshold: # should I trade only when the resi is above the transaction cost?
                     other_assets=self.data_returns.columns.drop(self.m_asset[today])
                     self.pos[self.m_asset[today]]=-np.sign(self.m_resi[today]) # we take a position on the asset
                     self.pos[other_assets]=np.sign(self.m_resi[today])*models[self.m_asset[today]].coef_ # we build the arbitrage portfolio
                     self.holding_period=0
                     self.entry_prices=self.data_prices.loc[today,:] # we record the entry prices
                     self.active_pos=True
                     if verbose: print('{}: Position open'.format(today))
            else: # in case of an active position we just wait for the mean reversion
                self.holding_period += 1
                if self.holding_period>=self.max_holding_period: # we close after a few periods, and record the pnl of the trade
                    self.ret[today]=sum(self.pos*(self.data_prices.loc[today,:]/self.entry_prices-1))-abs(self.pos.sum())*self.transaction_cost
                    self.pos[:]=0.0
                    self.active_pos=False
                    if verbose: print('{}: Position closed, Return: {:f}'.format(today, self.ret[today]))
        if verbose: print('Strategy calculated')
    
    def validation(self, hp_grid, verbose=False):
        ''' Validation step ''' 
        self.best_hp={}
        self.scores={}
        self.best_score_valid=-np.Inf
        self.compute_resi(self.rolling_window_size, self.validation_set_size, verbose)
        for hp in ParameterGrid(hp_grid):
            self.reinit()
            if verbose: print('Threshold:{}, Max Holding Period:{}'.format(hp['threshold'], hp['max_holding_period']))
            self.set_hp(**hp)
            self.compute_strat(self.rolling_window_size, self.validation_set_size, verbose)
            score=self.ret.sum()
            self.scores[(hp['threshold'], hp['max_holding_period'])]=score # we record each score
            if score>self.best_score_valid:
                self.best_score_valid=score
                self.best_hp=hp
        self.set_hp(**self.best_hp)

    def testing(self, include_val_period=False, verbose=False):
        ''' Testing step '''
        self.reinit()
        self.set_hp(**self.best_hp)        
        begin = self.rolling_window_size if include_val_period else self.validation_set_size
        self.compute_resi(self.rolling_window_size, len(self.data_returns.index), verbose)
        self.compute_strat(self.rolling_window_size, len(self.data_returns.index), verbose)

    def set_hp(self, **hp):
        ''' Used to change the value of the hyperparameters '''
        for name in hp:
            setattr(self, name, hp[name])
 
    def plot_cumret(self):
        ''' Used to print the cumulated returns '''
        self.compute_outputs()
        plot([Scattergl(x=self.cumret.index,y=self.cumret)]) # We use plotly to plot

    def compute_outputs(self):
        ''' Produce a set of outputs '''
        self.cumret=self.ret.cumsum()
        self.beta={}
        self.alpha={}
        for asset in self.data_returns.columns:    
            self.beta[asset]=self.ret.cov(pd.to_numeric(self.data_returns[asset]))
            self.alpha[asset]=np.mean(self.ret)-self.beta[asset]*np.mean(self.data_returns[asset]) # anualized?
        nb_y=(self.ret.index[-1]-self.ret.index[0])/datetime.timedelta(days=252)
        self.vol=np.std(self.ret)*np.sqrt(252*720) # annualized vol
        self.sharpe=self.cumret[-1]/(self.vol*nb_y) # annualized sharpe
        #self.max_drawdown=None

TS=TradingStrategy(models, data_returns, data_prices, rolling_window_size, validation_set_size, transaction_cost)

# Validation
TS.validation(hp_grid, verbose=1)

# Best hyperparameters
TS.best_hp    

# Testing
TS.testing(verbose=1)

##########################################################################################################
### OUTPUTS & STAT TESTS ###
##########################################################################################################

### Before checking the PnL let us check some of our assumptions

# The most important is the mean reverting behavior of our residuals
resi={}
adf_res={}
kpss_res={}
arima_res={}
acf_res={}
pacf_res={}

# Stationarity tests
for asset in data_returns.columns:
    try:
        adf_res[asset]=adfuller(TS.norm_resi[asset].dropna())         
    except:
        adf_res[asset]=[np.nan for _ in range(5)]
    try:
        kpss_res[asset]=kpss(TS.norm_resi[asset].dropna())
    except:
        kpss_res[asset]=[np.nan for _ in range(5)]

pvalues_adf=[adf_res[asset][1] for asset in data_returns.columns] # very low p values for ADF
pvalues_kpss=[kpss_res[asset][1] for asset in data_returns.columns] # very high (max is 0.1) p values for KPSS
# The test seems to confirm  the stationarity of this time serie


# ACF and PACF to check the mean reverting effect
for asset in data_returns.columns:
   acf_res[asset]=acf(pd.to_numeric(TS.norm_resi[asset].dropna()),qstat=True)

# We look at the p values for the first and second coefficiant of the ljung box test
# If the p value is small then we have a significative first and second autocorrelation
# This would mean that we have a very short term momentum or mean reverting in our residuals
pvalues_ljung_box_first=[(acf_res[asset][2][0]) for asset in data_returns.columns]
pvalues_ljung_box_second=[(acf_res[asset][2][1]) for asset in data_returns.columns]
# We see that some p values are high but most of them are very low, we do have an effect to capture here
sign_first_acf=[(np.sign(acf_res[asset][0][1])) for asset in data_returns.columns]
# The sign of the first acf coef is almost always negative: we have a mean reverting effect on the first period
sign_second_acf=[(np.sign(acf_res[asset][0][2])) for asset in data_returns.columns]
# looking at the next coefficiants shows us that actually the mean reverting effect last a few periods (around 5/6)
# but it is much more unclear, we need to make sure to close very soon the positions




### PnL reporting and risk measure
TS.plot_cumret()
TS.vol


pass  # debug




