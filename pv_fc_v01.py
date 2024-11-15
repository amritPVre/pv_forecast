import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Streamlit app
st.title('Solar Irradiance Forecasting with LSTM')
st.write("This app forecasts the next 24 hours of solar irradiance for a single location.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Display the data
    st.write("Dataset Preview:")
    st.write(data.head())
    
    # Assuming the dataset has a 'datetime' and 'solar_irradiance' column
    # Parse the datetime and set it as index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    
    # Visualize the solar irradiance data
    st.line_chart(data['solar_irradiance'])
    
    # Preprocessing
    st.write("Preprocessing the data...")
    values = data['solar_irradiance'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    
    # Create time-series windows for LSTM input (e.g., past 24 hours -> next hour)
    def create_dataset(dataset, look_back=24):
        X, y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i + look_back), 0])
            y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(y)

    look_back = 24  # Looking back 24 hours to predict the next one
    X, y = create_dataset(scaled_values, look_back)
    
    # Reshape X to be [samples, time steps, features] for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into training and test sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build the LSTM model
    st.write("Building the LSTM model...")
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model
    st.write("Training the model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2, validation_data=(X_test, y_test))
    
    # Plot training loss
    st.write("Training Loss vs. Validation Loss")
    loss_history = pd.DataFrame({
        "Training Loss": history.history['loss'],
        "Validation Loss": history.history['val_loss']
    })
    st.line_chart(loss_history)
    
    # Forecast the next 24 hours
    st.write("Forecasting the next 24 hours...")
    last_24_hours = scaled_values[-look_back:].reshape(1, look_back, 1)
    forecast_scaled = model.predict(last_24_hours)
    
    # Inverse transform to get actual values
    forecast = scaler.inverse_transform(forecast_scaled)
    
    # Display the forecasted values
    st.write(f"The forecasted solar irradiance for the next hour is: {forecast[0][0]}")
    
    # Repeat the process to get next 24 hours
    forecasts = []
    for i in range(24):
        next_24_hours = scaled_values[-look_back:].reshape(1, look_back, 1)
        forecast_scaled = model.predict(next_24_hours)
        forecast = scaler.inverse_transform(forecast_scaled)
        forecasts.append(forecast[0][0])
        scaled_values = np.append(scaled_values, forecast_scaled).reshape(-1, 1)

    # Plot the forecasted values
    st.write("Next 24 hours forecasted solar irradiance:")
    forecast_df = pd.DataFrame({
        'Hour': range(1, 25),
        'Forecasted Solar Irradiance': forecasts
    })
    
    st.line_chart(forecast_df.set_index('Hour'))
    
    st.write(forecast_df)

else:
    st.write("Please upload your dataset to proceed.")
