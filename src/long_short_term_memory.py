import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


class LongShortTermMemory:
    NUMBER_OF_FEATURE = 1

    def __init__(self, stock_ticker, start_date, end_date, training_duration):
        self.df = web.DataReader(stock_ticker, data_source='yahoo', start=start_date, end=end_date)
        self.data_set = self.df.filter(['Close']).values
        self.training_data_len = math.ceil(len(self.data_set) * 0.8)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.training_duration = training_duration
        self.scaled_data = self.scaler.fit_transform(self.data_set)

    def visualize(self, title, x_label, y_label):
        plt.figure(figsize=(16, 8))
        plt.title(title)
        plt.plot(self.df['Close'])
        plt.xlabel(x_label, fontsize=18)
        plt.ylabel(y_label, fontsize=18)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    def run_prediction(self, show_data_graph):
        if show_data_graph:
            self.visualize('Close Price History', 'Date', 'Close Price USD ($)')
        y_eval = self.data_set[self.training_data_len:, :]
        trained_model = self.training(self.scaled_data[0:self.training_data_len, :])
        predictions = self.evaluating(
            self.scaled_data[self.training_data_len - self.training_duration:, :],
            trained_model, y_eval)
        self.plot_model(predictions)

    def training(self, train_data):
        x_train = []
        y_train = []
        for i in range(self.training_duration, len(train_data)):
            x_train.append(train_data[i - self.training_duration:i, 0])
            y_train.append(train_data[i, 0])
            if i <= self.training_duration:
                print(x_train)
                print(y_train)
                print()

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], LongShortTermMemory.NUMBER_OF_FEATURE))
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

    def evaluating(self, eval_data, model, y_test):
        # Create the data set
        x_test = []
        for i in range(self.training_duration, len(eval_data)):
            x_test.append(eval_data[i - self.training_duration:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], LongShortTermMemory.NUMBER_OF_FEATURE))

        # Get the models predicted price values
        predictions = model.predict(x_test)
        # Contains the same value as y_test
        predictions = self.scaler.inverse_transform(predictions)

        # Get the RMSE, standard deviation of residual
        rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
        print(rmse)
        return predictions

    def plot_model(self, predicted_data):
        # Plot the data
        data = self.df.filter(['Close'])
        train = data[:self.training_data_len]
        valid = data[self.training_data_len:]
        valid['Predictions'] = predicted_data
        print(valid.shape)
        plt.figure(figsize=(16, 8))
        plt.title('Model Prediction')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Stock Price', 'Actual', 'Predictions'], loc='lower right')
        plt.show()
