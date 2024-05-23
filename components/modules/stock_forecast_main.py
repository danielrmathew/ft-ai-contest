#!/usr/bin/env python
# coding: utf-8

# In[3]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid


# In[14]:


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
    data['ds'] = data['Date'].dt.tz_localize(None)  # Convert Date to ds and remove timezone info
    data.rename(columns={'Close': 'y'}, inplace=True)  # Rename Close to y for Prophet
    data = data[['ds', 'y']]
    data.drop_duplicates(subset=['ds'], inplace=True)  # Remove duplicate dates
    return data

#-------------------------------------------------------------------------------------------------

def model_param_evaluation(data, params):
    """
    Evaluates Prophet model parameters using cross-validation.
    
    :param data: DataFrame, historical stock data
    :param params: dict, model parameters to test
    :return: float, mean RMSE from cross-validation
    """
    # Initialize Prophet model with given parameters
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        changepoint_range=params['changepoint_range'],
        holidays_prior_scale=25,
        seasonality_prior_scale=10,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    model.add_country_holidays(country_name='US')  # Add US holidays to model
    model.fit(data)  # Fit model to data
    
    # Perform cross-validation
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    df_p = performance_metrics(df_cv)
    return df_p['rmse'].mean()  # Return mean RMSE

#-------------------------------------------------------------------------------------------------

def hyperparameter_tuning(data, param_grid):
    """
    Performs hyperparameter tuning to find the best parameters for the Prophet model.
    
    :param data: DataFrame, historical stock data
    :param param_grid: dict, grid of parameters to search
    :return: dict, best parameters found
    """
    grid = ParameterGrid(param_grid)
    best_params = None
    best_rmse = float('inf')
    
    # Iterate over each combination of parameters
    for params in grid:
        rmse = model_param_evaluation(data, params)
        
        # Update best parameters if current RMSE is lower
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    
    print(f"Best parameters: {best_params}, Best RMSE: {best_rmse}")
    return best_params

#-------------------------------------------------------------------------------------------------

def train_prophet_model(data):
    """
    Trains a Prophet model using the best hyperparameters found.
    
    :param data: DataFrame, historical stock data
    :return: Prophet model trained with the best parameters
    """
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.015, 0.020, 0.025],
        'changepoint_range': [0.2, 0.4, 0.5, 0.8]
    }
    
    # Find best parameters
    best_params = hyperparameter_tuning(data, param_grid)

    # Initialize and train the Prophet model with best parameters
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

#-------------------------------------------------------------------------------------------------

def gen_stock_forecast(symbol, start_date, end_date, num_days):
    """
    Generates a stock price forecast for a given number of days.
    
    :param symbol: str, stock symbol
    :param start_date: str, start date (YYYY-MM-DD format)
    :param end_date: str, end date (YYYY-MM-DD format)
    :param num_days: int, number of days to forecast
    :return: trained Prophet model, DataFrame with forecast, full forecast DataFrame, trend value
    """
    # Fetch historical stock data
    data = fetch_stock_data(symbol, start_date, end_date)

    # Train the Prophet model
    model = train_prophet_model(data)

    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=num_days)

    # Predict future values
    forecast = model.predict(future)
    
    # Calculate the average of the last 'num_days' actual values
    prev_avg = data.tail(num_days)['y'].mean()
    # Calculate the average of the forecasted values
    next_avg = forecast['yhat'].mean()
    
    # Calculate the trend as the average of the previous and next average values
    trend = (prev_avg + next_avg) / 2
#     model.plot(forecast.to_numpy())

    return model, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num_days), forecast, trend


# In[15]:


def main_forecast(ticker, start_date, end_date, days_to_predict):
    """
    Generates a stock price forecast and returns the forecast trend and average confidence interval length.
    
    :param ticker: str, stock symbol
    :param start_date: str, start date (YYYY-MM-DD format)
    :param end_date: str, end date (YYYY-MM-DD format)
    :param days_to_predict: int, number of days to forecast
    :return: tuple, (forecast trend, average confidence interval length)
    """
    # Generate stock forecast using the specified parameters
    model, fc, fc_all, fc_trend = gen_stock_forecast(ticker, start_date, end_date, days_to_predict)
    
    # Calculate the predicted date 92 days after the end date
    pred_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=92)
    pred_date = pred_date.strftime('%Y-%m-%d')
    
    # Calculate the average confidence interval length
    cl = (fc['yhat_upper'] - fc['yhat_lower']).mean()
    
    return fc_trend, cl


# In[16]:


def stock_options_weights(tickers, start_date, end_date, days_to_predict):
    """
    Calculates the forecast trends and weights for a list of stock tickers based on their predicted trends and confidence intervals.
    
    :param tickers: list of str, stock symbols
    :param start_date: str, start date (YYYY-MM-DD format)
    :param end_date: str, end date (YYYY-MM-DD format)
    :param days_to_predict: int, number of days to forecast
    :return: dict, mapping each ticker to its forecast trend and calculated weight
    """
    
    def calculate_weights(xs):
        """
        Calculates weights based on the inverse of each value in xs.
        
        :param xs: array-like, values to calculate weights for
        :return: array, calculated weights
        """
        # Calculate the inverse of each value, adding a small constant to avoid division by zero
        inverses = 1 / (xs + 1e-10)

        # Sum of inverses
        total_inverse = sum(inverses)

        # Calculate weights as the proportion of each inverse over the total inverse
        weights = inverses / total_inverse

        return weights
    
    trends_cls = []
    
    # Loop through each ticker to calculate its trend and confidence interval length
    for ticker in tickers:
        trends_cls.append(main_forecast(ticker, start_date, end_date, days_to_predict))
        
    # Convert list to numpy array for easier manipulation
    trends_cls = np.array(trends_cls)
    
    # Extract trends and confidence interval lengths
    trends = trends_cls[:, 0]
    cls = trends_cls[:, 1]

    # Calculate weights based on confidence interval lengths
    weights = calculate_weights(cls)

    # Create a dictionary mapping each ticker to its trend and weight
    return {tickers[i]: (trends[i], weights[i]) for i in range(len(tickers))}


# In[11]:

"""
TESTING STOCK FORECAST BELOW
"""

# tick1 = 'AAPL'
# tick2 = 'AMZN'
# tick3 = 'TSLA'
# tickers = [tick1, tick2, tick3]
# start = '2020-01-01'
# end = '2024-01-01'
# days_to_predict = 92

# stock_prediction = stock_options_weights(tickers, start, end, days_to_predict)

# print(stock_prediction)

