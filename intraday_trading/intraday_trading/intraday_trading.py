import numpy as np
import pandas as pd
import os
import datetime
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV 
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
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
            data = data.iloc[-(n_max + 1):]     
       
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
        if i.date != last_timestamp.date: # If we change the day
            factor=res.loc[last_timestamp,:] # then we take as factor the last closing price
            last_timestamp=i
        res.loc[i,:]=df.loc[i,:]*factor # we multiply by the last closing price
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

stock_data,index_data=dataset_building(verbose=1)
stock_prices=compute_prices(stock_data)
index_prices=compute_prices(index_data)
stock_returns=compute_returns(stock_prices) # This compute the 2 min return
index_returns=compute_returns(index_prices)

data_returns=stock_returns.join(index_returns, how='outer')
data_prices=stock_prices.join(index_prices, how='outer')
data_returns.dropna(inplace=True) # This deletes a lot of data, we should find a clever way to deal with closed markets or restrict the number of assets
data_prices.dropna(inplace=True)

##########################################################################################################
### MODEL ###
##########################################################################################################

# we build a model for each asset
lr={asset:LinearRegression() for asset in data_prices.columns}
lasso={asset:Lasso() for asset in data_prices.columns}


models=lr

##########################################################################################################
### GLOBAL LOOK AT THE DATA ###
##########################################################################################################

# We will test our hypothesis about the market behavior by
# building a system of linear models on 

n_testing_days=10
last_sample_day=data_returns.index[0]+datetime.timedelta(days=n_testing_days) 
last_sample_day_index=data_returns.index.get_loc(last_sample_day, method='ffill')
sample_index=data_returns.index[:last_sample_day_index]

# We will store the redisuals, and results to ouor tests
resi={}
adf_res={}
kpss_res={}
arima_res={}
acf_res={}
pacf_res={}

for asset in data_returns.columns:
    other_assets=data_returns.columns.drop(asset)
    Y=data_returns.loc[sample_index,asset]
    X=data_returns.loc[sample_index, other_assets]
    lr[asset].fit(X, Y) # we fit the model
    resi[asset]=Y-lr[asset].predict(X) # compute the residuals
    adf_res[asset]=adfuller(resi[asset])   # we first test for the stationarity of the residuals       
    kpss_res[asset]=kpss(resi[asset])     # we test for the mean reverting behavior of the residuals

pvalues_adf=[adf_res[asset][1] for asset in data_returns.columns] # very low p values for ADF
pvalues_kpss=[kpss_res[asset][1] for asset in data_returns.columns] # very high (max is 0.1) p values for KPSS
# The tests confirm the stationary hypothesis on the residuals

for asset in data_returns.columns:
   # we have stationarity, so we know d=0 in our arima, to find p and d we need to look at the ACF and PACF
   acf_res[asset]=acf(resi[asset],qstat=True)
   pacf_res[asset]=pacf(resi[asset])

# We look at the p values  for the first coefficiant of the ljung box test
# If the p value s small then we have a significative first autocorrelation
# This would mean that we have a very short term momentum or mean reverting on our residuals
pvalues_ljung_box_first=[(acf_res[asset][2][0]) for asset in data_returns.columns]
pvalues_ljung_box_second=[(acf_res[asset][2][1]) for asset in data_returns.columns]
# We see that some p values are high but most of them are very low, we do have an effect to capture here
sign_first_acf=[(np.sign(acf_res[asset][0][1])) for asset in data_returns.columns]
# The sign of the first acf coef is always negative: we have a mean reverting effect on the first period
sign_second_acf=[(np.sign(acf_res[asset][0][2])) for asset in data_returns.columns]
# looking at the next coefficiants shows us that actually the mean reverting effect last a few periods (around 5/6)
# but it is much more unclear, we need to make sure to close very soon the positions

# ARIMA?????
   #arima[asset]=ARIMA(resi, order=(5,0,5)) # d=0 because we have stationarity
   

# Of course there tests are based on a sample and should be taken carefully
# In order to check if this strategy could really make money we need to code it properly by simulating 
# a trading environment with no look ahead bias

##########################################################################################################
### TRADING STRATEGY ###
##########################################################################################################

# We will work with a rolling window in the following way:
# Every 2 min we will look at the residuals of our assets versus our predicted assets
# the prediction will be based on the linear models built using the information from the rolling window

rolling_window_size=256 # The number of data points we need to build our models, 256 for one day

resi={}
norm_resi={} # The norm residuals are the residuals normalized by their std dev
pnl=pd.Series(0,index=data_returns.index) # keep track of the realized pnl
pos=pd.DataFrame(0,index=data_returns.index, columns=data_prices.columns) # keep track of the positions

active_pos=False # this will help us keep track of our open strategy, and make sure we have only one strategy running at the same time

# Hyperparameter: the limit on z score of the residual after which we trade
threshold=2
max_holding_period=5 # we define a max holding period

# information will keep track of during the strategy
holding_period=0
entry_prices=pd.DataFrame()

for i in range(rolling_window_size,len(data_returns.index)):
    today=data_returns.index[i]
    print(today) # debug
    if i > 1000:
        break
    if not active_pos:
        train=data_returns.index[i-rolling_window_size:i-1]
        for asset in data_returns.columns:  
            other_assets=data_returns.columns.drop(asset)
            Y=data_returns.loc[train,asset]
            X=data_returns.loc[train,other_assets]
            models[asset].fit(X, Y)
            resi=Y-models[asset].predict(X) # These are the residuals
            norm_resi=resi[-1]/np.std(resi) # This is our last residual 
            if abs(norm_resi)>threshold:
                pos.loc[today,asset]=-np.sign(norm_resi) # we take a position on the asset
                for other_asset in other_assets:
                    j=other_assets.get_loc(other_asset)
                    pos.loc[today,other_asset]=np.sign(norm_resi)*models[asset].coef_[j] # we build the arbitrage portfolio
                holding_period=0
                entry_prices=data_prices.loc[today,:] # we record the entry prices
                active_pos=True
                print('position open')
                break # we stop looping through assets if we take a position
    else: # in case of an active position we just wait for the mean reversion
        holding_period += 1
        pos.iloc[i,:]=pos.iloc[i-1,:]
        if holding_period>=max_holding_period: # we close after a few periods, and record the pnl of the trade
            pnl[today]=sum(pos.loc[today,:]*(data_prices.loc[today,:]-entry_prices))
            pos.loc[today,:]=0
            active_pos=False
            print('position closed')


pass  # debug




