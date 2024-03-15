import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model  
import streamlit as st 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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

# Load data for the selected stock
def load_stock_data(stock_ticker, start_date, end_date):
    df = yf.download(stock_ticker, start=start_date, end=end_date)
    return df['Close']

# Streamlit app
def main():
    st.title("Stock Price Trend Prediction") 

    # User input for stock ticker
    user_input = st.text_input("Enter any Stock's Ticker Symbol", 'MSFT')
    if not user_input:
        st.warning("Please enter a valid stock ticker symbol.")
        return
    stock_ticker = user_input.upper()  # Convert to uppercase for consistency

    # Load data for the selected stock
    try:
        df = load_stock_data(stock_ticker, '2014-01-01', '2023-12-31')
    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        return

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

    # Other visualizations and model predictions...
    
if __name__ == "__main__":
    main()
