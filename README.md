# Stock-Market-Prediction-using-LSTM-and-ARIMA
This repository features a hybrid stock price prediction model using ARIMA for short-term forecasting and LSTM for modeling residuals. It includes data preprocessing, stationarity checks, and performance evaluation, demonstrating effective machine learning techniques for financial forecasting.
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Fetch stock data using yfinance
def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data['Close']  # We focus on the "Close" price for prediction

# Step 1: Fetch the stock data
stock_symbol = 'AAPL'  # Example: Apple Inc.
start_date = '2015-01-01'
end_date = '2023-01-01'

stock_data = fetch_stock_data(stock_symbol, start_date, end_date)

# Step 2: Plot the closing stock price data
plt.figure(figsize=(10,6))
plt.plot(stock_data)
plt.title(f'{stock_symbol} Stock Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# Step 3: Make the series stationary (ARIMA requirement)
# Apply differencing to make the data stationary
stock_data_diff = stock_data.diff().dropna()

# Plot ACF and PACF to determine ARIMA parameters
plt.figure(figsize=(12,6))
plot_acf(stock_data_diff)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(stock_data_diff)
plt.show()

# Based on ACF and PACF plots, choose ARIMA parameters (p, d, q)
# Example: p=5, d=1, q=0 (This is just an example; it should be optimized based on your data)
p, d, q = 5, 1, 0

# Step 4: Fit the ARIMA model
arima_model = ARIMA(stock_data, order=(p, d, q))
arima_result = arima_model.fit()

# Step 5: Plot ARIMA predictions (train predictions)
arima_pred = arima_result.predict(start=0, end=len(stock_data)-1, typ='levels')

# Plot the original and ARIMA predicted values
plt.figure(figsize=(10,6))
plt.plot(stock_data, label='Actual')
plt.plot(arima_pred, label='ARIMA Predictions')
plt.legend()
plt.title('ARIMA Stock Price Prediction')
plt.show()

# Step 6: Get residuals from ARIMA model
residuals = stock_data - arima_pred

# Step 7: Scale the residuals for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# Step 8: Prepare the data for LSTM
# Create a dataset where X is the input sequence and y is the output
def create_lstm_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # Number of previous days to use as input for prediction
X_residual, y_residual = create_lstm_dataset(residuals_scaled, time_step)

# Reshape data for LSTM (samples, time steps, features)
X_residual = X_residual.reshape(X_residual.shape[0], X_residual.shape[1], 1)

# Step 9: Build and train the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_residual.shape[1], 1)))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dense(units=1))  # Output layer

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_residual, y_residual, epochs=10, batch_size=32)

# Step 10: Make predictions using LSTM model
lstm_pred_scaled = lstm_model.predict(X_residual)

# Step 11: Inverse transform LSTM predictions back to the original scale
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

# Step 12: Combine ARIMA and LSTM predictions
# Add the ARIMA predictions and LSTM residual predictions
combined_pred = arima_pred + lstm_pred.flatten()

# Step 13: Plot the final combined predictions vs actual stock price
plt.figure(figsize=(10,6))
plt.plot(stock_data, label='Actual')
plt.plot(combined_pred, label='Combined ARIMA + LSTM Predictions')
plt.legend()
plt.title('Stock Price Prediction (ARIMA + LSTM)')
plt.show()

# Step 14: Evaluate the model
mse = mean_squared_error(stock_data[time_step:], combined_pred[time_step:])
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

Code Explanation:
Importing Libraries: Necessary libraries such as numpy, pandas, matplotlib, yfinance, statsmodels, and tensorflow are imported for data processing, modeling, and visualization.

Fetching Data: Stock data is fetched from Yahoo Finance using yfinance for a specific stock symbol (e.g., Apple Inc.). The closing price is used for prediction.

Plotting Stock Price: The historical closing stock prices are plotted to visualize the data.

Differencing for Stationarity: ARIMA requires the data to be stationary, so the data is differenced (subtracted from previous values) to make it stationary. ACF and PACF plots are used to determine ARIMA parameters (p, d, q).

Fitting ARIMA Model: The ARIMA model is fitted to the stationary stock data with the determined parameters (p, d, q).

ARIMA Prediction: The ARIMA model is used to predict stock prices, and these predictions are plotted alongside the actual stock prices for comparison.

Residuals from ARIMA: The residuals (errors) are calculated as the difference between actual and ARIMA-predicted values. These residuals are then scaled to fit the LSTM model.

Preparing Data for LSTM: Data is prepared for LSTM by creating sequences of residual values (sliding windows). The LSTM model will learn to predict residuals.

Building LSTM Model: An LSTM model is created with two layers of 50 units each and a final dense layer for output. The model is compiled and trained.

LSTM Prediction: LSTM makes predictions on the residuals, which are then inverse-transformed back to the original scale.

Combining Predictions: The final stock price prediction is obtained by adding the ARIMA predictions and the LSTM predictions (on residuals).

Plotting Final Results: The combined predictions are plotted against actual stock prices for comparison.

Evaluation: The model's performance is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

Notes:
The p, d, and q parameters for the ARIMA model are chosen based on ACF and PACF plots, but these should be optimized for better performance.
The time step for the LSTM model (time_step=60) can be adjusted to suit the nature of the stock data.
This is a simple implementation; you can experiment with hyperparameter tuning for better performance.
