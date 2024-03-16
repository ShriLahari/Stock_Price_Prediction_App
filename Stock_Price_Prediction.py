 # importing all the necessary libraries to do the project
import tensorflow 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
# Creating a Deep Learning model using keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# defining the start and end date of stock prices
start_date ='2014-01-01'
end_date ='2023-12-31'

st.title("Stock Price Trend Prediction")

user_input = st.text_input("Enter any Stock's Ticker Symbol",'MSFT')
df = yf.download("MSFT",start_date,end_date)

# Statistical data analysis
st.subheader("Summary Statistical data of stocks from 2014 to 2023")
st.write(df.describe())

# Visualisations
st.subheader("Closing Price vs Time graph")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, color='black', label='Closing Price')
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.legend()
st.pyplot(fig)

# Visualisations
st.subheader("Closing Price vs Time graph with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, color='blue', label='100MA')
plt.plot(df.Close, color='black', label='Closing Price')
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.legend()
st.pyplot(fig)

# Visualisations
st.subheader("Closing Price vs Time graph with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, color='blue', label='100MA')
plt.plot(ma200, color='red', label='200MA')
plt.plot(df.Close, color='black', label='Closing Price')
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.legend()
st.pyplot(fig)

# Splitting the data into training and testing (train:70, test:30)

df_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
df_test =  pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
# Scaling the data using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

# Scaling the training and testing data separately
mms_train = MinMaxScaler(feature_range=(0, 1))
mms_test = MinMaxScaler(feature_range=(0, 1))

df_train_scaled = mms_train.fit_transform(df_train)
df_test_scaled = mms_test.fit_transform(df_test)

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
Stock_model.fit(X_train, y_train, epochs=50, batch_size=32)

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
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'r', label='Original values')
plt.plot(y_predict,'green',label="Predicted values")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# metrics
st.subheader("Metrics to evaluate the performance of the model:")

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Calculate MAE
mae = mean_absolute_error(y_test, y_predict)
st.write("Mean Absolute Error(MAE):", mae)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_predict)
st.write("R-squared (R2) Score:", r2)

# Calculate Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_test, y_predict, squared=False)
st.write("Root Mean Squared Error (RMSE):", rmse)
