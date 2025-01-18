# Stock-Market-Prediction-using-LSTM-and-ARIMA
This repository features a hybrid stock price prediction model using ARIMA for short-term forecasting and LSTM for modeling residuals. It includes data preprocessing, stationarity checks, and performance evaluation, demonstrating effective machine learning techniques for financial forecasting.
Table of Contents
Installation
Project Overview
Dataset
Usage
Model Explanation
ARIMA Model
LSTM Model
Results
Evaluation
Contributing
License
Installation
To run this project, you need to install the required dependencies. You can install them using pip:

bash
Copy
Edit
pip install -r requirements.txt
The requirements.txt includes the following packages:

numpy
pandas
matplotlib
scikit-learn
tensorflow
statsmodels
yfinance
Project Overview
This project predicts stock prices using a hybrid approach that integrates two popular models:

ARIMA (AutoRegressive Integrated Moving Average): Used for capturing short-term, linear trends.
LSTM (Long Short-Term Memory): A deep learning model used to capture complex, non-linear dependencies and trends.
The combination of ARIMA and LSTM enhances the accuracy of the predictions by addressing both linear and non-linear aspects of stock price movements.

Dataset
The dataset used for this project is the historical stock price data of a selected company. The data is fetched using the yfinance library, which allows for easy downloading of stock data. The primary focus is on the "Close" price of the stock.

Data Columns:
Date: The date of the stock price.
Open: Opening price of the stock on that day.
High: Highest price of the stock during that day.
Low: Lowest price of the stock during that day.
Close: Closing price of the stock on that day.
Volume: The volume of stocks traded.
Usage
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
Download and Prepare Data: Use yfinance to fetch stock data and prepare it for modeling.

Preprocess the Data:

Perform data normalization using MinMaxScaler.
Apply differencing for ARIMA to make the series stationary.
Build the Models:

ARIMA model is trained on the stationary data.
LSTM model is trained on ARIMA residuals.
Make Predictions: Generate stock price predictions using the combined model and evaluate the results.

Visualize Results: Plot the actual vs predicted stock prices.

Model Explanation
ARIMA Model
ARIMA is used to model the linear patterns and trends in the stock price data. The model is trained on the differenced data to make the series stationary. The parameters of the ARIMA model (p, d, q) are chosen based on the ACF (AutoCorrelation Function) and PACF (Partial AutoCorrelation Function) plots.

LSTM Model
LSTM is used to capture complex, non-linear relationships in the residuals obtained from the ARIMA model. The residuals are passed through an LSTM network, which consists of multiple LSTM layers and dense layers, to make predictions on future residuals. These predictions are then added to the ARIMA predictions to get the final stock price forecast.

Results
The model predicts future stock prices by combining the predictions of the ARIMA and LSTM models. The results show how well the hybrid model performs in comparison to a single ARIMA or LSTM model.

Evaluation
The performance of the model is evaluated using the following metrics:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
These metrics help assess how close the predictions are to the actual values and the overall accuracy of the model.

Contributing
Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests. Please follow the standard GitHub workflows for contributions.

