import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

st.title("Stock Price Trend Prediction")

# User input for stock symbol
user_input = st.text_input("Enter any Stock's Ticker Symbol", 'MSFT')

# Downloading data
start_date = '2014-01-01'
end_date = '2023-12-31'
df = yf.download(user_input, start=start_date, end=end_date)

if df.empty:
    st.error("No data found for the provided stock symbol. Please enter a valid symbol.")
else:
    # Statistical data analysis
    st.subheader("Summary Statistical data of stocks from 2014 to 2023")
    st.write(df.describe())

    # Visualizations
    st.subheader("Closing Price vs Time graph")
    st.line_chart(df['Close'])

    st.subheader("Closing Price vs Time graph with 100MA")
    ma100 = df['Close'].rolling(100).mean()
    plt.plot(ma100, color='blue', label='100MA')
    plt.plot(df['Close'], color='black', label='Closing Price')
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot()

    st.subheader("Closing Price vs Time graph with 100MA & 200MA")
    ma200 = df['Close'].rolling(200).mean()
    plt.plot(ma100, color='blue', label='100MA')
    plt.plot(ma200, color='red', label='200MA')
    plt.plot(df['Close'], color='black', label='Closing Price')
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot()

    # Splitting the data into training and testing (train:70, test:30)
    df_train = df['Close'].iloc[:int(len(df) * 0.7)]
    df_test = df['Close'].iloc[int(len(df) * 0.7):]

    # Scaling the data using MinMaxScaler
    mms_train = MinMaxScaler(feature_range=(0, 1))
    mms_test = MinMaxScaler(feature_range=(0, 1))

    df_train_scaled = mms_train.fit_transform(df_train.values.reshape(-1, 1))
    df_test_scaled = mms_test.fit_transform(df_test.values.reshape(-1, 1))

    # Prepare X_train, y_train
    X_train, y_train = [], []
    for i in range(100, len(df_train_scaled)):
        X_train.append(df_train_scaled[i - 100:i, 0])
        y_train.append(df_train_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Model Architecture
    Stock_model = Sequential()
    Stock_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    Stock_model.add(Dropout(0.2))
    Stock_model.add(LSTM(units=60, return_sequences=True))
    Stock_model.add(Dropout(0.3))
    Stock_model.add(LSTM(units=80, return_sequences=True))
    Stock_model.add(Dropout(0.4))
    Stock_model.add(LSTM(units=50))
    Stock_model.add(Dropout(0.5))
    Stock_model.add(Dense(units=1))

    # Compiling the model
    Stock_model.compile(optimizer='adam', loss='mean_squared_error')

    # Training the model
    Stock_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Prepare test data
    X_test, y_test = [], []
    for i in range(100, len(df_test_scaled)):
        X_test.append(df_test_scaled[i - 100:i, 0])
        y_test.append(df_test_scaled[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predictions
    y_predict = Stock_model.predict(X_test)

    # Inverse scaling
    y_predict = mms_test.inverse_transform(y_predict)
    y_test = mms_test.inverse_transform(y_test.reshape(-1, 1))

    # Visualisations
    st.subheader("Comparing the Actual prices with Predicted prices")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'r', label='Original values')
    plt.plot(y_predict, 'green', label="Predicted values")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig2)

    # metrics
    st.subheader("Metrics to evaluate the performance of the model:")
    mae = mean_absolute_error(y_test, y_predict)
    st.write("Mean Absolute Error(MAE):", mae)

    r2 = r2_score(y_test, y_predict)
    st.write("R-squared (R2) Score:", r2)

    rmse = mean_squared_error(y_test, y_predict, squared=False)
    st.write("Root Mean Squared Error (RMSE):", rmse)
