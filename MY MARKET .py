# # # # import pandas as pd
# # # # import numpy as np
# # # # from sklearn.linear_model import LinearRegression
# # # # from sklearn.preprocessing import MinMaxScaler

# # # # # Load your CSV file
# # # # data = pd.read_csv('POWERGRID.NS.csv')

# # # # # Use the 'Close' prices for prediction
# # # # close_prices = data['Close'].values.reshape(-1, 1)

# # # # # Normalize the data for better performance
# # # # scaler = MinMaxScaler(feature_range=(0, 1))
# # # # scaled_data = scaler.fit_transform(close_prices)

# # # # # Preparing the features for linear regression
# # # # X_linreg = np.arange(len(scaled_data)).reshape(-1, 1)  # Use the time index as a feature

# # # # # Train the model on the full dataset
# # # # model_linreg = LinearRegression()
# # # # model_linreg.fit(X_linreg, scaled_data)

# # # # # Predict the price for August 15, 2024 (assuming it's the next time point)
# # # # next_day_index = len(scaled_data)
# # # # predicted_price_scaled = model_linreg.predict([[next_day_index]])

# # # # # Convert the predicted scaled value back to the original price
# # # # predicted_price = scaler.inverse_transform(predicted_price_scaled)

# # # # print(f"Predicted price for 2024-08-15: {predicted_price[0][0]:.2f}")
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.linear_model import LinearRegression
# # # from sklearn.preprocessing import MinMaxScaler

# # # # Load your CSV file
# # # data = pd.read_csv('POWERGRID.NS.csv')

# # # # Use the 'Close' prices for prediction
# # # close_prices = data['Close'].values.reshape(-1, 1)

# # # # Normalize the data for better performance
# # # scaler = MinMaxScaler(feature_range=(0, 1))
# # # scaled_data = scaler.fit_transform(close_prices)

# # # # Preparing the features for linear regression
# # # X_linreg = np.arange(len(scaled_data)).reshape(-1, 1)  # Use the time index as a feature

# # # # Train the model on the full dataset
# # # model_linreg = LinearRegression()
# # # model_linreg.fit(X_linreg, scaled_data)

# # # # Predict prices for the next N days (e.g., 30 days)
# # # future_days = 30
# # # predicted_prices = []

# # # for i in range(1, future_days + 1):
# # #     next_day_index = len(scaled_data) + i - 1
# # #     predicted_price_scaled = model_linreg.predict([[next_day_index]])
# # #     predicted_price = scaler.inverse_transform(predicted_price_scaled)
# # #     predicted_prices.append(predicted_price[0][0])

# # # # Find the lowest (bad) and highest (good) predicted prices
# # # lowest_price = min(predicted_prices)
# # # highest_price = max(predicted_prices)

# # # print(f"Predicted prices for the next {future_days} days:")
# # # print(f"Highest predicted price: {highest_price:.2f}")
# # # print(f"Lowest predicted price: {lowest_price:.2f}")

# # # # Print the predicted prices for each day (optional)
# # # for i, price in enumerate(predicted_prices, 1):
# # #     print(f"Day {i}: {price:.2f}")

# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import MinMaxScaler
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, Dense, Dropout

# # # Load your CSV file
# # data = pd.read_csv('POWERGRID.NS.csv')

# # # Use the 'Close' prices for prediction
# # close_prices = data['Close'].values.reshape(-1, 1)

# # # Normalize the data
# # scaler = MinMaxScaler(feature_range=(0, 1))
# # scaled_data = scaler.fit_transform(close_prices)

# # # Create sequences of 60 days for prediction
# # def create_sequences(data, sequence_length):
# #     sequences = []
# #     labels = []
# #     for i in range(len(data) - sequence_length):
# #         sequences.append(data[i:i + sequence_length])
# #         labels.append(data[i + sequence_length])
# #     return np.array(sequences), np.array(labels)

# # sequence_length = 60
# # X, y = create_sequences(scaled_data, sequence_length)

# # # Split the data into training and test sets (optional)
# # split = int(0.8 * len(X))
# # X_train, y_train = X[:split], y[:split]
# # X_test, y_test = X[split:], y[split:]

# # # Build the LSTM model
# # model = Sequential()
# # model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# # model.add(Dropout(0.2))
# # model.add(LSTM(units=50, return_sequences=False))
# # model.add(Dropout(0.2))
# # model.add(Dense(units=25))
# # model.add(Dense(units=1))

# # # Compile the model
# # model.compile(optimizer='adam', loss='mean_squared_error')

# # # Train the model
# # model.fit(X_train, y_train, batch_size=32, epochs=20)

# # # Predict future prices for the next N days (e.g., 30 days)
# # future_days = 30
# # predicted_prices = []

# # last_sequence = scaled_data[-sequence_length:]  # Start with the last sequence in the training data

# # for i in range(future_days):
# #     prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
# #     predicted_prices.append(scaler.inverse_transform(prediction)[0][0])

# #     # Append the prediction to the last_sequence and remove the oldest value
# #     last_sequence = np.append(last_sequence[1:], prediction)

# # # Find the lowest and highest predicted prices
# # lowest_price = min(predicted_prices)
# # highest_price = max(predicted_prices)

# # print(f"Predicted prices for the next {future_days} days:")
# # print(f"Highest predicted price: {highest_price:.2f}")
# # print(f"Lowest predicted price: {lowest_price:.2f}")

# # # Print the predicted prices for each day (optional)
# # for i, price in enumerate(predicted_prices, 1):
# #     print(f"Day {i}: {price:.2f}")

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from datetime import timedelta

# # Load your CSV file
# data = pd.read_csv('POWERGRID.NS.csv')

# # Example: Add a 'NewsImpact' column (binary or sentiment score)
# # Replace this with your actual data source for news impact
# np.random.seed(42)
# data['NewsImpact'] = np.random.randint(0, 2, size=len(data))

# # Use 'Close' prices and 'NewsImpact' for prediction
# close_prices = data['Close'].values.reshape(-1, 1)
# news_impact = data['NewsImpact'].values.reshape(-1, 1)

# # Normalize the price data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_close_prices = scaler.fit_transform(close_prices)

# # Combine the scaled close prices with the news impact data
# combined_data = np.hstack((scaled_close_prices, news_impact))

# # Create sequences of 60 days for prediction
# def create_sequences(data, sequence_length):
#     sequences = []
#     labels = []
#     for i in range(len(data) - sequence_length):
#         sequences.append(data[i:i + sequence_length])
#         labels.append(data[i + sequence_length, 0])  # Predicting the price
#     return np.array(sequences), np.array(labels)

# sequence_length = 60
# X, y = create_sequences(combined_data, sequence_length)

# # Split the data into training and test sets (optional)
# split = int(0.8 * len(X))
# X_train, y_train = X[:split], y[:split]
# X_test, y_test = X[split:], y[split:]

# # Build the LSTM model
# input_shape = (X_train.shape[1], X_train.shape[2])

# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(units=25))
# model.add(Dense(units=1))  # Predicting the price

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X_train, y_train, batch_size=32, epochs=20)

# # Predict future prices for the next N days (e.g., 30 days)
# future_days = 30
# predicted_prices = []

# last_sequence = combined_data[-sequence_length:]  # Start with the last sequence in the training data

# # Get the last date from the original data
# last_date = pd.to_datetime(data['Date'].iloc[-1])

# # Store dates for predictions
# future_dates = []

# for i in range(future_days):
#     prediction = model.predict(last_sequence.reshape(1, sequence_length, 2))
#     predicted_prices.append(scaler.inverse_transform(prediction)[0][0])

#     # Append the prediction to the last_sequence and remove the oldest value
#     new_news_impact = np.random.randint(0, 2)  # Replace with actual news impact
#     new_entry = np.array([[prediction[0][0], new_news_impact]])
#     last_sequence = np.append(last_sequence[1:], new_entry, axis=0)

#     # Add the next date
#     future_dates.append(last_date + timedelta(days=i + 1))

# # Find the lowest and highest predicted prices
# lowest_price = min(predicted_prices)
# highest_price = max(predicted_prices)

# print(f"Predicted prices for the next {future_days} days:")
# print(f"Highest predicted price: {highest_price:.2f}")
# print(f"Lowest predicted price: {lowest_price:.2f}")

# # Print the predicted prices for each day with the date
# for date, price in zip(future_dates, predicted_prices):
#     print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta

# Load your CSV file
data = pd.read_csv('POWERGRID.NS.csv')

# Example: Add a 'NewsImpact' column (binary or sentiment score)
# Replace this with your actual data source for news impact
np.random.seed(42)
data['NewsImpact'] = np.random.randint(0, 2, size=len(data))

# Use 'Close' prices and 'NewsImpact' for prediction
close_prices = data['Close'].values.reshape(-1, 1)
news_impact = data['NewsImpact'].values.reshape(-1, 1)

# Normalize the price data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close_prices = scaler.fit_transform(close_prices)

# Combine the scaled close prices with the news impact data
combined_data = np.hstack((scaled_close_prices, news_impact))

# Create sequences of 60 days for prediction
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length, 0])  # Predicting the price
    return np.array(sequences), np.array(labels)

sequence_length = 60
X, y = create_sequences(combined_data, sequence_length)

# Split the data into training and test sets (optional)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Build the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Predicting the price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=20)

# Predict future prices for the next N days (e.g., 30 days)
future_days = 30
predicted_prices = []

last_sequence = combined_data[-sequence_length:]  # Start with the last sequence in the training data

# Get the last date from the original data
last_date = pd.to_datetime(data['Date'].iloc[-1])

# Store dates for predictions
future_dates = []

for i in range(future_days):
    daily_predictions = []
    # Run multiple predictions for each day to simulate variability (optional)
    for _ in range(10):  # Simulate 10 possible outcomes for the day
        prediction = model.predict(last_sequence.reshape(1, sequence_length, 2))
        daily_predictions.append(scaler.inverse_transform(prediction)[0][0])

        # Create new sequence with the predicted price and random news impact
        new_news_impact = np.random.randint(0, 2)  # Replace with actual news impact
        new_entry = np.array([[prediction[0][0], new_news_impact]])
        last_sequence = np.append(last_sequence[1:], new_entry, axis=0)

    # Get the lowest and highest prices for that day
    day_lowest_price = min(daily_predictions)
    day_highest_price = max(daily_predictions)
    predicted_prices.append((day_lowest_price, day_highest_price))

    # Add the next date
    future_dates.append(last_date + timedelta(days=i + 1))

# Print the predicted prices for each day with the date
for date, (low_price, high_price) in zip(future_dates, predicted_prices):
    print(f"{date.strftime('%Y-%m-%d')} - Lowest: {low_price:.2f}, Highest: {high_price:.2f}")


