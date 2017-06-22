import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge 
from sklearn.model_selection import ParameterGrid
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


# The code is quite long to run, especially the part that computes residuals
# Do the analysis on a reduced dataset to test it faster, ex: n_max=15000
stock_data,index_data=dataset_building(n_max=None,verbose=1)
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
# since we normalized the input we can let alpha=1  for lasso, the best would be to cross validate it
# but it would be to heavy in terms of computations
models=lasso
 
##########################################################################################################
### TRADING STRATEGY ###
##########################################################################################################

# We will work with a rolling window in the following way:
# Every 2 min we will look at the residuals of our assets versus our predicted assets
# the prediction will be based on the linear models built using the information from the rolling window

# Global params
rolling_window_size=720 # The number of data points we need to build our models, 720 for one day
data_size=len(data_returns.index)
validation_set_size=int(data_size/4)
testing_set_size=data_size-validation_set_size


# We define a class to keep track of our trading strategy
class TradingStrategy():
    ''' This class will help us keep track of our trading strategy '''
    def __init__(self, models, data_returns, data_prices, rolling_window_size, validation_set_size):
        self.models=models
        self.data_returns=data_returns
        self.data_prices=data_prices
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
            now=self.data_returns.index[i]
            #if verbose: print(now) # debug
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
            self.m_resi[now]=max_resi
            self.m_asset[now]=max_asset
        if verbose: print('Residuals calculated')        
     
    def compute_strat(self, begin, end, verbose=False, delta_time_check=False):
        ''' This function will compute the strategy on a certain period, we need to compute the residuals first '''
        if verbose: print('Calculating Strategy...')
        for i in range(begin, end):     
            now=self.data_returns.index[i]   
            if not self.active_pos: # we look for a signal on our max residual
                 if delta_time_check:
                    try: # the try block allow us to catch the out of bound exception
                        time_check=self.m_resi.index[i+self.max_holding_period]-now<self.max_holding_period*datetime.timedelta(minutes=2)
                    except IndexError:
                        time_check=False
                 else:
                    time_check=True
                 if abs(self.m_resi[now])>self.threshold and time_check:
                     other_assets=self.data_returns.columns.drop(self.m_asset[now])
                     self.pos[self.m_asset[now]]=-np.sign(self.m_resi[now]) # we take a position on the asset
                     self.pos[other_assets]=np.sign(self.m_resi[now])*models[self.m_asset[now]].coef_ # we build the arbitrage portfolio
                     self.holding_period=0
                     self.entry_prices=self.data_prices.loc[now,:] # we record the entry prices
                     self.active_pos=True
                     if verbose: print('{}: Position open'.format(now))
            else: # in case of an active position we just wait for the mean reversion
                self.holding_period += 1
                if self.holding_period>=self.max_holding_period: # we close after a few periods, and record the pnl of the trade
                    self.ret[now]=sum(self.pos*(self.data_prices.loc[now,:]/self.entry_prices-1))-abs(self.pos.sum())*self.transaction_cost
                    self.pos[:]=0.0
                    self.active_pos=False
                    if verbose: print('{}: Position closed, Return: {:f}'.format(now, self.ret[now]))
        if verbose: print('Strategy calculated')
    
    def validation(self, hp_grid, transaction_cost=0.0, verbose=False, delta_time_check=False):
        ''' Validation step,  please compute residuals before ''' 
        self.best_hp={}
        self.scores={}
        self.best_score_valid=-np.Inf
        self.transaction_cost=transaction_cost
        for hp in ParameterGrid(hp_grid):
            self.reinit()
            if verbose: print('Threshold:{}, Max Holding Period:{}'.format(hp['threshold'], hp['max_holding_period']))
            self.set_hp(**hp)
            self.compute_strat(self.rolling_window_size, self.validation_set_size, verbose, delta_time_check)
            score=self.ret.sum()
            self.scores[(hp['threshold'], hp['max_holding_period'])]=score # we record each score
            if score>self.best_score_valid:
                self.best_score_valid=score
                self.best_hp=hp
        
    def testing(self, include_val_period=False, transaction_cost=0.0, verbose=False, delta_time_check=False):
        ''' Testing step, please compute residuals before '''
        self.reinit()
        self.set_hp(**self.best_hp)  
        self.transaction_cost=transaction_cost      
        begin = self.rolling_window_size if include_val_period else self.validation_set_size
        self.compute_strat(begin, len(self.data_returns.index), verbose, delta_time_check)
        self.compute_outputs(include_val_period)

    def valid_test(self, hp_grid, include_val_period=False, verbose=False):
        ''' Validation & Testing '''
        self.validation(hp_grid, verbose)
        self.testing(include_val_period, verbose)

    def set_hp(self, **hp):
        ''' Used to change the value of the hyperparameters '''
        for name in hp: setattr(self, name, hp[name])
 
    def plot_cumret(self):
        ''' Used to print the cumulated returns '''
        plot([Scattergl(x=self.cumret.index,y=self.cumret)]) # We use plotly to plot

    def compute_outputs(self, include_val_period=True):
        ''' Produce a set of outputs '''
        self.cumret=self.ret.cumsum()
        self.beta={}
        self.alpha={}
        for asset in self.data_returns.columns:    
            self.beta[asset]=self.ret.cov(pd.to_numeric(self.data_returns[asset]))
            self.alpha[asset]=np.mean(self.ret)-self.beta[asset]*np.mean(self.data_returns[asset]) # anualized?
        begin = self.validation_set_size if not include_val_period else 0
        nb_y=(self.ret.index[-1]-self.ret.index[begin])/datetime.timedelta(days=365)
        self.vol=np.std(self.ret)*np.sqrt(252*720) # annualized vol
        self.ann_cumret=self.cumret[-1]/nb_y
        self.sharpe=self.ann_cumret/self.vol 
        #self.max_drawdown=None



TS=TradingStrategy(models, data_returns, data_prices, rolling_window_size, validation_set_size)

# Compute the residuals
TS.compute_resi(TS.rolling_window_size, len(TS.data_returns.index), verbose=1)

# Hyperparameters grid & Transaction costs: 
hp_grid={'threshold':np.linspace(1,10,19), # the limit on z score of the residual after which we trade
         'max_holding_period':[1,2,5,10,15,30,60,240,480]} # holding period of the positions
transaction_cost=0.00075


# Validation
TS.validation(hp_grid, transaction_cost, verbose=1, delta_time_check=True)

# Best hyperparameters
TS.best_hp    

# Testing
TS.testing(include_val_period=False, transaction_cost=transaction_cost, verbose=1, delta_time_check=True)

### PnL reporting and risk measure
TS.plot_cumret()
TS.ann_cumret
TS.vol
TS.sharpe



res['7.5bp']=TS.cumret

data_plot=[Scattergl(x=res.index,y=res['0bp'],name='0 bp'),
      Scattergl(x=res.index,y=res['2.5bp'],name='2.5 bps'),
      Scattergl(x=res.index,y=res['5bp'],name='5 bps'),
      Scattergl(x=res.index,y=res['7.5bp'],name='7.5 bps'),
      Scattergl(x=res.index,y=res['10bp'],name='10 bps'),
      Scattergl(x=res.index,y=res['15bp'],name='15 bps')]

plot(data_plot)  





