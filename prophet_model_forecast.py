#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install prophet


# In[5]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid


# In[6]:


def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches historical stock prices for a given symbol from start_date to end_date.
    Dates are exclusive
    
    :param symbol: str, stock symbol
    :param start_date: str, start date (YYYY-MM-DD format)
    :param end_date: str, end date (YYYY-MM-DD format)
    :return: DataFrame with historical stock data
    """
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    data.reset_index(inplace=True) 
    data['ds'] = data['Date'].dt.tz_localize(None)
    data.rename(columns={'Close': 'y'}, inplace=True)
    data = data[['ds', 'y']]
    data.drop_duplicates(subset=['ds'], inplace=True)
    return data


def model_param_evaluation(data, params):
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        changepoint_range=params['changepoint_range'],
        holidays_prior_scale=25,
        seasonality_prior_scale=10,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    model.add_country_holidays(country_name='US')
    model.fit(data)
    
    # Perform cross-validation
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
    df_p = performance_metrics(df_cv)
    return df_p['rmse'].mean()


def hyperparameter_tuning(data, param_grid):
    grid = ParameterGrid(param_grid)
    best_params = None
    best_rmse = float('inf')
    
    for params in grid:
        rmse = model_param_evaluation(data, params)
#         print(f"Tested params: {params}, RMSE: {rmse}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    
    print(f"Best parameters: {best_params}, Best RMSE: {best_rmse}")
    return best_params



def train_prophet_model(data):
    param_grid = {
    'changepoint_prior_scale': [0.01, 0.015, 0.020, 0.025],
    'changepoint_range': [0.2, 0.4, 0.5, 0.8]

    }
    
    best_params = hyperparameter_tuning(data, param_grid)

    model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        changepoint_range=best_params['changepoint_range'],
        holidays_prior_scale=25,
        seasonality_prior_scale=10,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    
    model.add_country_holidays(country_name='US')
    model.fit(data)
    return model

def train_prophet_original_model(data):

    model = Prophet(
        changepoint_prior_scale=0.02,
        changepoint_range=0.95,
        holidays_prior_scale=25,
        seasonality_prior_scale=10,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    
    model.add_country_holidays(country_name='US')
    model.fit(data)
    return model    
    

def gen_stock_forecast(symbol, start_date, end_date, num_days):
    # Fetch data
    data = fetch_stock_data(symbol, start_date, end_date)

    # Train Prophet model
    model = train_prophet_model(data)

    # Make future dataframe for prediction
    future = model.make_future_dataframe(periods=num_days)

    # Predict future values
    forecast = model.predict(future)
    
    prev_avg = data.tail(num_days)['y'].mean()
    next_avg = forecast['yhat'].mean()
    
    trend = (prev_avg + next_avg) / 2
#     model.plot(forecast.to_numpy())

    return model, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num_days), forecast, trend

def gen_stock_forecast_original(symbol, start_date, end_date, num_days):
    # Fetch data
    data = fetch_stock_data(symbol, start_date, end_date)

    # Train Prophet model
    model = train_prophet_original_model(data)

    # Make future dataframe for prediction
    future = model.make_future_dataframe(periods=num_days)

    # Predict future values
    forecast = model.predict(future)
    
    prev_avg = data.tail(num_days)['y'].mean()
    next_avg = forecast['yhat'].mean()
    
    trend = (prev_avg + next_avg) / 2
#     model.plot(forecast.to_numpy())

    return model, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num_days), forecast, trend


    


# In[5]:


# sym = 'AMZN'
# start = '2020-01-01'
# end = '2024-01-01'
# num_days = 92

# model, amzn_fc, amzn_all, amzn_trend = gen_stock_forecast_original(sym, start, end, num_days)
# amzn_true = fetch_stock_data('AMZN', '2024-01-01', '2024-05-01')

# plt.figure(figsize=(10, 5))


# plt.plot(np.array(amzn_fc['ds']), np.array(amzn_fc['yhat']), label='Forecast', color='blue')
# plt.plot(np.array(amzn_true['ds']), np.array(amzn_true['y']), label='True Y', color='green')

# plt.title('Stock Price Forecast')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()


# In[14]:


# sym = 'AMZN'
# start = '2020-01-01'
# end = '2024-01-01'
# num_days = 100

# model, amzn_fc, amzn_all, amzn_trend = gen_stock_forecast(sym, start, end, num_days)
# amzn_true = fetch_stock_data('AMZN', '2024-01-01', '2024-05-01')

# plt.figure(figsize=(10, 5))


# plt.plot(np.array(amzn_fc['ds']), np.array(amzn_fc['yhat']), label='Forecast', color='blue')
# plt.plot(np.array(amzn_true['ds']), np.array(amzn_true['y']), label='True Y', color='green')

# plt.title('Stock Price Forecast')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()


# In[6]:


# sym = 'AAPL'
# start = '2020-01-01'
# end = '2024-01-01'
# num_days = 92

# model, aapl_fc, aapl_all, aapl_trend = gen_stock_forecast(sym, start, end, num_days)
# aapl_true = fetch_stock_data('AAPL', '2024-01-01', '2024-05-01')

# plt.figure(figsize=(10, 5))
# plt.plot(np.array(aapl_fc['ds']), np.array(aapl_fc['yhat']), label='Forecast', color='blue')
# plt.plot(np.array(aapl_true['ds']), np.array(aapl_true['y']), label='True Y', color='green')

# plt.title('Stock Price Forecast')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()


# In[7]:


# sym = 'AAPL'
# start = '2020-01-01'
# end = '2024-01-01'
# num_days = 92

# model, aapl_fc, aapl_all, aapl_trend = gen_stock_forecast_original(sym, start, end, num_days)
# aapl_true = fetch_stock_data('AAPL', '2024-01-01', '2024-05-01')

# plt.figure(figsize=(10, 5))
# plt.plot(np.array(aapl_fc['ds']), np.array(aapl_fc['yhat']), label='Forecast', color='blue')
# plt.plot(np.array(aapl_true['ds']), np.array(aapl_true['y']), label='True Y', color='green')

# plt.title('Stock Price Forecast')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()


# In[7]:


# amzn_trend


# In[10]:


# aapl_trend
# aapl_fc


# In[7]:


def main_forecast(ticker, start_date, end_date, days_to_predict):
    model, fc, fc_all, fc_trend = gen_stock_forecast(ticker, start_date, end_date, days_to_predict)
    
    pred_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=92)
    pred_date = pred_date.strftime('%Y-%m-%d')
    
    cl = (fc['yhat_upper'] - fc['yhat_lower']).mean()
    
    
#     fc_true = fetch_stock_data(ticker, end_date, pred_date)

    
#     plt.figure(figsize=(10, 5))
#     plt.plot(np.array(fc['ds']), np.array(fc['yhat']), label='Forecast', color='blue')
#     plt.plot(np.array(fc_true['ds']), np.array(fc_true['y']), label='True Y', color='green')

#     plt.title(f'{ticker} Price Forecast')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()
    
    # smaller confidence level range means greater confidence
    return fc_trend, cl


# In[8]:


def stock_options_weights(tickers, start_date, end_date, days_to_predict):
    def calculate_weights(xs):
        # Calculate the inverse of each value
        inverses = 1 / (xs + 1e-10)
#         for x in xs:
#             inverses.append(1 / (x + 1e-10)) # Adding a small epsilon to avoid division by zero
#         inverse1 = 1 / (x1 + 1e-10)  
#         inverse2 = 1 / (x2 + 1e-10)
#         inverse3 = 1 / (x3 + 1e-10)

        # Sum of inverses
        total_inverse = sum(inverses)

        # Calculate weights as the proportion of each inverse to the total sum of inverses
#         weight1 = inverse1 / total_inverse
#         weight2 = inverse2 / total_inverse
#         weight3 = inverse3 / total_inverse
        weights = inverses / total_inverse

        # Scale weights to sum to 100
#         scaled_weights = weights * 100

        return weights
    
    trends_cls = []
    for ticker in tickers:
        trends_cls.append(main_forecast(ticker, start_date, end_date, days_to_predict))
    trends_cls = np.array(trends_cls)
    
#     trend_1, cl_1 = main_forecast(tick1, start_date, end_date, days_to_predict)
#     trend_2, cl_2 = main_forecast(tick2, start_date, end_date, days_to_predict)
#     trend_3, cl_3 = main_forecast(tick3, start_date, end_date, days_to_predict)
    
    trends = trends_cls[:, 0]
    cls = trends_cls[:, 1]
    weights = calculate_weights(cls)
#     return {tick1: (trend_1, weights[0]), tick2: (trend_2, weights[1]), tick3: (trend_3, weights[2])}

    return {tickers[i]: (trends[i], weights[i]) for i in range(len(tickers))}
    
    


# In[9]:


# tick1 = 'AAPL'
# tick2 = 'AMZN'
# tick3 = 'TSLA'
# start = '2020-01-01'
# end = '2024-01-01'
# days_to_predict = 92

# stock_prediction = stock_options_weights(tick1, tick2, tick3, start, end, days_to_predict)

# print(stock_prediction)


# In[13]:


# ticker = 'AAPL'
# start = '2020-01-01'
# end = '2024-01-01'
# days_to_predict = 92

# predicted_trend, predicted_cl = main_forecast(ticker, start, end, days_to_predict)

# print(f'{ticker} forecasted trend: {predicted_trend}')


# In[14]:


# sym = 'AAPL'
# start = '2020-01-01'
# end = '2024-01-01'
# num_days = 92

# model, fc_apple, fc_all, fc_trend = gen_stock_forecast(sym, start, end, num_days)


# # In[15]:


# fc_cl = (fc_apple['yhat_upper'] - fc_apple['yhat_lower']).mean()
# fc_cl


# # In[16]:


# sym = 'AMZN'
# start = '2020-01-01'
# end = '2024-01-01'
# num_days = 92

# model, fc_amzn, fc_all, fc_trend = gen_stock_forecast(sym, start, end, num_days)


# # In[17]:


# fc_cl2 = (fc_amzn['yhat_upper'] - fc_amzn['yhat_lower']).mean()
# fc_cl2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


# def num_to_day(i):
#     dct = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
#     return dct[i]
# def date_to_day(dt):
#     month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
#     day = ''
    
# #     if dt.dayofyear <= 9:
# #         day = '0' + str(dt.dayofyear)
# #     else:
# #         day = str(dt.dayofyear)
    
#     day = str(dt.day)
        
#     return month[dt.month - 1] + ' ' + day
# amzn_all['date'] = pd.to_datetime(amzn_all['ds'])
# amzn_all['day_of_week'] = amzn_all['date'].dt.dayofweek
# amzn_all['day_of_week'] = amzn_all['day_of_week'].apply(num_to_day)
# amzn_all['day_of_year'] = amzn_all['date'].apply(date_to_day)

# amzn_all.head()


# In[5]:


# plt.figure(1, figsize=(12, 5))
# plt.plot(np.array(amzn_all['ds']), np.array(amzn_all['trend']), label='trend vs ds', color='blue')

# plt.figure(2, figsize=(12, 5))
# temp1 = amzn_all.groupby('day_of_week', sort=False).mean(numeric_only=True)['weekly']
# plt.plot(np.array(temp1.index), np.array(temp1.values), label='day_of_week', color='green')

# plt.figure(3, figsize=(20, 5))
# temp2 = amzn_all.groupby('day_of_year', sort=False).mean(numeric_only=True)['yearly']
# upd_labs = []
# for dt in np.array(temp2.index):
#     labels = ['January 1', 'February 1', 'March 1', 'April 1', 'May 1', 'June 1', 'July 1', 'August 1', 'September 1', 'October 1', 'November 1', 'December 1']
#     if dt in labels:
#         upd_labs.append(dt)
#     else:
#         upd_labs.append('')
# plt.plot(np.array(temp2.index), np.array(temp2.values), label='day_of_year', color='red')
# plt.xticks(np.array(temp2.index), upd_labs, rotation=45, ha='right')

# plt.figure(3, figsize=(12, 5))
# plt.plot(np.array(amzn_all['ds']), np.array(amzn_all['daily']), label='trend vs ds', color='blue')
pass


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[218]:


# import itertools

# param_grid = {
#     'changepoint_prior_scale': [0.01, 0.05, 0.1],
#     'holidays_prior_scale': [5, 10, 15],
#     'seasonality_prior_scale': [5, 10, 15],
#     'weekly_seasonality': [True, False],
#     'yearly_seasonality': [True, False],
#     'daily_seasonality': [True, False]
# }

# all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
# rmses = []  

# for params in all_params:
#     m = Prophet(**params).fit(df)  # Fit model with given params
#     df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
#     df_p = performance_metrics(df_cv, rolling_window=1)
#     rmses.append(df_p['rmse'].values[0])

# tuning_results = pd.DataFrame(all_params)
# tuning_results['rmse'] = rmses
# print(tuning_results)


# In[ ]:





# In[ ]:




