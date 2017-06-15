import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV 
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
        res[col] = ((1+df.loc[:,col])/(1+df.loc[:,col].shift(window))) - 1
    return res

stock_data,index_data=dataset_building(n_max=10000,verbose=1)
#data=stock_data.join(index_data,how='outer')
data=index_data
data.dropna(inplace=True)
data_prices=compute_prices(data).apply(pd.to_numeric)
data_returns=compute_returns(data_prices)
data_returns.dropna(inplace=True)

##########################################################################################################
### MODEL ###
##########################################################################################################

# we build a model for each asset
lr={asset:LinearRegression() for asset in data_prices.columns}
lasso={asset:Lasso() for asset in data_prices.columns}


models=lr

##########################################################################################################
### TRADING STRATEGY ###
##########################################################################################################

# We will work with a rolling window in the following way:
# Every 2 min we will look at the residuals of our assets versus our predicted assets
# the prediction will be based on the linear models built using the information from the rolling window


# Global params
rolling_window_size=720 # The number of data points we need to build our models, 720 for one day
transaction_cost=0.05 # in returns for a trade (entry and exit included)
data_size=len(data_returns.index)
validation_set_size=int(data_size/4)
testing_set_size=data_size-validation_set_size


# Hyperparameters grid: 
threshold_grid=[2,3,4,5] # the limit on z score of the residual after which we trade
max_holding_period_grid=[1,5,10,15] # holding period of the positions

# We define a class to keep track of our trading strategy
class trading_strategy():
    ''' This class will help us keep track of our trading strategy '''
    def __init__(self, models, data_returns, data_prices, rolling_window_size, transaction_cost):
        self.models=models
        self.data_returns=data_returns
        self.data_prices=data_prices
        self.transaction_cost=transaction_cost
        self.rolling_window_size=rolling_window_size
        self.norm_resi={} # The norm residuals are the residuals normalized by their std dev
        self.ret=pd.Series(0.0,index=data_returns.index) # keep track of the realized pnl
        self.pos=pd.DataFrame(0.0,index=data_returns.index, columns=data_prices.columns) # keep track of the positions
        self.norm_resi=pd.DataFrame(index=data_returns.index, columns=data_prices.columns) # nomalized residuals
        self.active_pos=False # this will help us keep track of our open strategy, and make sure we have only one strategy running at the same time
        self.holding_period=0
        self.entry_prices=pd.DataFrame()
        self.best_hp={}
        self.best_score_valid=-np.Inf

        # Hyperparameters
        self.threshold=None
        self.max_holding_period=None

    def reinit(self):
        self.norm_resi={} # The norm residuals are the residuals normalized by their std dev
        self.ret=pd.Series(0.0,index=data_returns.index) # keep track of the realized pnl
        self.pos=pd.DataFrame(0.0,index=data_returns.index, columns=data_prices.columns) # keep track of the positions
        self.norm_resi=pd.DataFrame(index=data_returns.index, columns=data_prices.columns) # nomalized residuals
        self.active_pos=False # this will help us keep track of our open strategy, and make sure we have only one strategy running at the same time
        self.holding_period=0
        self.entry_prices=pd.DataFrame()
        
    def one_period(self, i):
        ''' This function will operate one trading period '''
        today=self.data_returns.index[i]
        print(today) # debug
        if not self.active_pos:
            train=self.data_returns.index[i-self.rolling_window_size:i-1]
            for asset in self.data_returns.columns:  
                other_assets=self.data_returns.columns.drop(asset)
                Y=self.data_returns.loc[train,asset]
                X=self.data_returns.loc[train,other_assets]
                self.models[asset].fit(X, Y)
                resi=Y-self.models[asset].predict(X) # These are the residuals
                self.norm_resi.loc[today,asset]=resi[-1]/np.std(resi) # This is our last residual 
                if abs(self.norm_resi.loc[today,asset])>self.threshold:
                    self.pos.loc[today,asset]=-np.sign(self.norm_resi.loc[today,asset]) # we take a position on the asset
                    self.pos.loc[today,other_assets]=np.sign(self.norm_resi.loc[today,asset])*models[asset].coef_ # we build the arbitrage portfolio
                    self.holding_period=0
                    self.entry_prices=self.data_prices.loc[today,:] # we record the entry prices
                    self.active_pos=True
                    print('position open')
                    return
            return
        else: # in case of an active position we just wait for the mean reversion
            self.holding_period += 1
            self.pos.iloc[i,:]=self.pos.iloc[i-1,:]
            if self.holding_period>=self.max_holding_period: # we close after a few periods, and record the pnl of the trade
                self.ret[today]=sum(self.pos.loc[today,:]*((self.data_prices.loc[today,:]-self.entry_prices)/self.entry_prices-self.transaction_cost))
                self.pos.loc[today,:]=0.0
                self.active_pos=False
                print('position closed, Return: {:f}'.format(self.ret[today]))
            return
    
    def set_hp(self, **kwargs):
        ''' Used to change the value of the hyperparameters '''
        for name in kwargs:
            setattr(self,name, kwargs[name])
 
    def print_cumret(self):
        ''' Used to print the cumulated returns '''
        cumret=self.ret.cumsum()
        plot([Scattergl(x=cumret.index,y=cumret)])

TS=trading_strategy(models, data_returns, data_prices, rolling_window_size, transaction_cost)

# Validation
for threshold in threshold_grid:
    TS.reinit()
    for max_holding_period in max_holding_period_grid:
        print('Threshold:{}, Max Holding Period:{}'.format(threshold, max_holding_period))
        TS.set_hp(threshold=threshold, max_holding_period=max_holding_period)
        for i in range(rolling_window_size,validation_set_size):
            TS.one_period(i)
        score=TS.ret.sum()
        if score>TS.best_score_valid:
            TS.best_score_valid=score
            TS.best_hp={'threshold':threshold, 'max_holding_period':max_holding_period}
    
# Testing
TS.reinit()
TS.set_hp(*{TS.best_hp})
for i in range(rolling_window_size,len(data.returns)):
    TS.one_period(i)


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
    adf_res[asset]=adfuller(norm_resi[asset])         
    kpss_res[asset]=kpss(norm_resi[asset])

pvalues_adf=[adf_res[asset][1] for asset in data_returns.columns] # very low p values for ADF
pvalues_kpss=[kpss_res[asset][1] for asset in data_returns.columns] # very high (max is 0.1) p values for KPSS
# The test seems to confirm  the stationarity of this time serie


# ACF and PACF to check the mean reverting effect
for asset in data_returns.columns:
   acf_res[asset]=acf(norm_resi[asset],qstat=True)
   pacf_res[asset]=pacf(norm_resi[asset])

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




### PnL reporting
cumret=ret.cumsum()
plot([Scattergl(x=cumret.index,y=cumret)])



pass  # debug




