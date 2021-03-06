import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

NUMBER_OF_FEATURE = 1
TRAINING_DURATION = 60

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')


def visualize(graph_data_set, title, x_label, y_label):
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.plot(graph_data_set)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.show()


data_set = df.filter(['Close']).values
training_data_len = math.ceil(len(data_set) * 0.8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_set)
y_eval = data_set[training_data_len:, :]


def training(train_data):
    x_train = []
    y_train = []
    for i in range(TRAINING_DURATION, len(train_data)):
        x_train.append(train_data[i - TRAINING_DURATION:i, 0])
        y_train.append(train_data[i, 0])
        if i <= TRAINING_DURATION:
            print(x_train)
            print(y_train)
            print()

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], NUMBER_OF_FEATURE))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model


def evaluating(eval_data, model, y_test):
    # Create the data set
    x_test = []
    for i in range(TRAINING_DURATION, len(eval_data)):
        x_test.append(eval_data[i - TRAINING_DURATION:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], NUMBER_OF_FEATURE))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    # Contains the same value as y_test
    predictions = scaler.inverse_transform(predictions)

    # Get the RMSE, standard deviation of residual
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(rmse)
    return predictions


def plot_model(data, predicted_data):
    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predicted_data
    # Visualize the model
    plt.figure(figsize=(16, 8))
    plt.title('Model Prediction')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


# visualize(df['Close'], 'Close Price History', 'Date', 'Close Price USD ($)')
trained_model = training(scaled_data[0:training_data_len, :])
predictions = evaluating(scaled_data[training_data_len - TRAINING_DURATION:, :], trained_model, y_eval)
plot_model(df.filter(['Close']), predictions)
# # Get the quote
# apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
# # Create a new dataframe
# new_df = apple_quote.filter(['Close'])
# # Get the last 60 day closing price values and convert the dataframe to an array
# last_60_days = new_df[-60:].values
# # Scale the data to be values between 0 and 1
# last_60_days_scaled = scaler.transform(last_60_days)
# # Create an empty list
# X_test = []
# # Append the past 60 last_60_days
# X_test.append(last_60_days_scaled)
# # Convert the X_test data set to a numpy array
#
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], NUMBER_OF_FEATURE))
# pred_price = trained_model.predict(X_test)
# pred_price = scaler.inverse_transform(pred_price)
# print(pred_price)
#
# apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
# print(apple_quote2['Close'])
