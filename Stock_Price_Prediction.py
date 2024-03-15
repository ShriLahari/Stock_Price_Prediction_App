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
    df = load_stock_data(stock_ticker, '2014-01-01', '2023-12-31')

    # Print debug information
    st.write("Data loaded successfully.")
    st.write("Head of the data:")
    st.write(df.head())

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

    # Closing Price vs Time graph with 100-day Moving Average
    # Visualizations 
    st.subheader("Closing Price vs Time graph with 100MA")
    ma100 = df.rolling(100).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100, color='blue', label='100MA')
    plt.plot(df, color='black', label='Closing Price')
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(fig)

    # Closing Price vs Time graph with 100-day and 200-day Moving Averages
    # Visualizations 
    st.subheader("Closing Price vs Time graph with 100MA & 200MA")
    ma100 = df.rolling(100).mean()
    ma200 = df.rolling(200).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100, color='blue', label='100MA')
    plt.plot(ma200, color='red', label='200MA')
    plt.plot(df, color='black', label='Closing Price')
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(fig)

    # Split data into training and testing
    train_size = int(len(df) * 0.70)
    df_train, df_test = df[:train_size], df[train_size:]

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train.values.reshape(-1, 1))

    # Prepare training and testing data
    X_train, y_train = prepare_data(df_train, scaler)
    X_test, y_test = prepare_data(df_test, scaler)

    # Load model
    model = load_stock_model("Stock_Price_Model.keras")

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
