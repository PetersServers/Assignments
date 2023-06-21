import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dropout, Activation, Dense, LSTM
from keras.layers import Bidirectional
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from statistics import mean

#create a neural net
#should only be used for crypto

graphics = True

ticker = "BTC-USD"
train_period = "10wk"
interval = "1h"
training_seq_len = 10 #100
epochs = 10


#######################################################################################################################
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
#######################################################################################################################


def download_transform_data(ticker, train_period, interval):

    df = yf.download(tickers=ticker, period=train_period, interval=interval)
    df = df[["Close"]]
    scaler = MinMaxScaler()
    close_price = df.Close.values.reshape(-1, 1)
    scaled = scaler.fit_transform(close_price)
    scaled = scaled[~np.isnan(scaled)]
    scaled = scaled.reshape(-1, 1)

    return scaled, scaler


def to_sequences(data, seq_len):
    d = []
    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])
    return np.array(d)


def preprocess(data_raw, seq_len, train_split, predict=False):
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]
    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]
    if predict: return X_test, y_test
    else: return X_train, y_train, X_test, y_test

def create_model(X_train, y_train):
    BATCH_SIZE = 64
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=32, return_sequences=True,
                          input_shape=(99, 1), dropout=0.2))
    model.add(layers.LSTM(units=32, return_sequences=True,
                          dropout=0.2))
    model.add(layers.LSTM(units=32, dropout=0.2))
    model.add(layers.Dense(units=1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
                        shuffle=False, validation_split=0.1)
    model.summary()
    return model, history

def download_transform_latest(ticker=ticker, period=train_period, interval=interval):
    scaled_asset, scaler = download_transform_data(ticker, period, interval)
    X_test, y_test =  preprocess(scaled_asset, training_seq_len, train_split=0.80, predict=True)
    return X_test, y_test


def calculate_range(training_seq_len): #main function right now

    scaled_asset, scaler = download_transform_data(ticker, train_period, interval)
    X_train, y_train, X_test, y_test = preprocess(scaled_asset, training_seq_len, train_split=0.80)
    model, history = create_model(X_train, y_train)
    model.evaluate(X_test, y_test)

    #download the latest data and predict /is redundant in this format
    X_test, y_test = download_transform_latest()
    y_hat = model.predict(X_test)
    y_test_inverse = scaler.inverse_transform(y_test)
    y_hat_inverse = scaler.inverse_transform(y_hat)

    y_test_list_inv = [i for i in y_test_inverse[0]]
    y_hat_list_inv = [i for i in y_hat_inverse[0]]


    efficiency = mean(abs(x - y) for x, y in zip(y_test_list_inv, y_hat_list_inv))
    print(f"the efficency of test range {training_seq_len} amounts to {efficiency}".upper())

    if graphics:

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'model loss training seq {training_seq_len}')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(y_test_inverse, label="Actual Price", color ="green")
        plt.plot(y_hat_inverse, label="Predicted Price", color ="red")
        plt.suptitle(f'Bitcoin price prediction LSTM training seq {training_seq_len}')
        plt.title(f'Efficiency = {efficiency}')
        plt.xlabel('Time[days]')
        plt.ylabel('Price')
        plt.legend(loc='best')

        plt.show();

    return efficiency

def find_best_range():

    efficiency = {}
    for i in range(8, 15):
        efficiency[f"{i}"] = calculate_range(i)

    maximal_efficiency = min(efficiency, key=efficiency.get)

    return maximal_efficiency


print(find_best_range())
