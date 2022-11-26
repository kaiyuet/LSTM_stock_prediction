import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure



T = 10
D = 1

def load_data():
    df = pd.read_csv('data2.csv')
    return df 


def data_standardize(df):
    series = df['close'].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(series[:len(series) // 2])
    series = scaler.transform(series).flatten()
    return series


def build_data(series):
    
    X = []
    Y = []

    for t in range(len(series)-T):
        x = series[t: t+T]
        y = series[t+T]
        X.append(x)
        Y.append(y)

    X = np.array(X).reshape(-1, T, 1)
    Y = np.array(Y)
    return X, Y

def build_model():
    i = Input(shape=(T, 1))
    #x = LSTM(3, return_sequences = True)(i)
    x = LSTM(5, return_sequences = True)(i)
    x = LSTM(7)(x)
    x = Dense(1)(x)
    model = Model(i, x)
    model.compile(
        loss='mse', 
        optimizer=Adam(lr=0.1)
    )

    return model





if __name__ == "__main__":
    df = load_data()
    series = data_standardize(df)
    X, Y = build_data(series)
    model = build_model()
    
    N = len(X)
    r = model.fit(
    X[:N//2], Y[:N//2], 
    epochs=80, 
    validation_data=(X[-N//2:], Y[-N//2:])
    )


    output = model.predict(X[N//2:])
    output[:, 0]
    plt.plot(Y[N//2:], label='Real Price')
    plt.plot(output[:, 0], label='Predict')
    plt.legend()
    plt.show()
    print(np.sum((Y[N//2:] - output[:, 0]) * (Y[N//2:] - output[:, 0])))
    results = model.evaluate(X[:N//2], Y[:N//2] )
    plt.plot(r.history['val_loss'])
    trade = []
    for i in range(len(X)//2):
        if(X[len(X)//2 + i][9][0] < Y[len(X)//2 + i]):
            trade.append(1)
        else:
            trade.append(0)
    trade = np.array(trade)

    predict_trade = []
    for i in range(len(X)//2):
        if(X[len(X)//2 + i][9][0] < output[:, 0][i]):
            predict_trade.append(1)
        else: 
            predict_trade.append(0)

    count = 0
    for i in range(len(trade)):
        if(trade[i] == predict_trade[i]):
            count += 1
    
    print(count)
    




