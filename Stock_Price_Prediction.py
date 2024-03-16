import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model  
import streamlit as st 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import requests

# Define function to preprocess and prepare data
def prepare_data(df, scaler):
    # Scaling the data
    scaled_data = scaler.transform(df.values.reshape(-1, 1)).flatten()
    X, y = [], []
    for i in range(100, len(scaled_data)):
        X.append(scaled_data[i - 100:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# Load model
def load_stock_model(model_path):
    return load_model(model_path)

def load_stock_data(stock_ticker, start_date, end_date):
    df = yf.download(stock_ticker, start=start_date, end=end_date)
    return df['Close']

# Streamlit app
def main():
    st.title("Stock Price Trend Prediction") 

    # User input for stock ticker
    user_input = st.text_input("Enter any Stock's Ticker Symbol", 'MSFT')
    stock_ticker = user_input.upper()  # Convert to uppercase for consistency

    # Load data for the selected stock
    df = load_stock_data(stock_ticker, '2014-01-01', '2023-12-31')

    # Display summary statistical data
    st.subheader("Summary Statistics for stocks from 2014 to 2023")
    st.write(df.describe()) 

    # Closing Price vs Time graph
    # Visualizations 
    st.subheader("Closing Price vs Time graph")
    fig = plt.figure(figsize=(12,6))
    plt.plot(df, color='black', label='Closing Price')
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(fig)

    # Load model
    model_path = "Stock_Price_Model.h5"
    model = load_stock_model(model_path)

    # Split data into training and testing
    train_size = int(len(df) * 0.70)
    df_train, df_test = df[:train_size], df[train_size:]

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train.values.reshape(-1, 1))

    # Prepare training and testing data
    X_train, y_train = prepare_data(df_train, scaler)
    X_test, y_test = prepare_data(df_test, scaler)

    # Make predictions
    y_predict = model.predict(X_test)
    y_predict = scaler.inverse_transform(y_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    rmse = mean_squared_error(y_test, y_predict, squared=False)

    # Display metrics
    st.subheader("Metrics to evaluate the performance of the model:")
    st.write("Mean Absolute Error (MAE):", mae)
    st.write("R-squared (R2) Score:", r2)
    st.write("Root Mean Squared Error (RMSE):", rmse)

if __name__ == "__main__":
    main()
