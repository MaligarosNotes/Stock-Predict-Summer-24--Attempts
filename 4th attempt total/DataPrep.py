import numpy as np
import pandas as pd
import matplotlib as plt
import yfinance as yf
import keras 
import os
import datetime 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def addStdDev(df,IV=1,daysago=100):
    dfstddev = pd.DataFrame()
    for i in range(1,IV):
        dfstddev['1/{} of daysago# std'.format(i)] = df.iloc[:,daysago-(daysago//i):daysago+1].std(axis=1)
    sc = MinMaxScaler(feature_range=(0,1))
    stddev_scaled = sc.fit_transform(dfstddev)
    return stddev_scaled, dfstddev

def TickerIntoTensor(ticker='aapl',daysago=100,daysforward=1,split_percentage=0.9,IV=1):
    df = yf.Ticker(ticker.upper())
    df = df.history(period='max')
    if df.shape[1] == 7:
        df = df.drop(columns=['Dividends','Stock Splits'])
    df = df.drop(columns=['Open','High','Low','Volume'])
    for i in range(daysago,-daysforward-1,-1):
        df['{}daysago'.format(i)] = df['Close'].shift(i)
        df = df.copy()
    df = df.drop(columns=['Close'])
    df = df.drop(index=df.tail(daysforward).index)
    df = df.drop(index=df.head(daysago).index)



    if IV == 0:
        literally_useless = 0
    else:
        tempSTDDEV = addStdDev(df,IV,daysago)
        stddev_scaled = tempSTDDEV[0]
        dfstddev = tempSTDDEV[1]



    sc = MinMaxScaler(feature_range=(0,1))
    df_scaled = sc.fit_transform(df)
    X,Y = df_scaled[:,:daysago+1],df_scaled[:,-daysforward:]
    if split_percentage <= 1 and split_percentage >= 0:
        split = int(len(X)*split_percentage)
    else:
        split = split_percentage
    X_train, X_test, Y_train, Y_test = X[:split], X[split:], Y[:split], Y[split:]
    X_train, X_test = np.append(stddev_scaled[:split],X_train,axis=1), np.append(stddev_scaled[split:],X_test,axis=1)
    X_train, X_test, Y_train, Y_test = X_train.reshape((-1,daysago+1+dfstddev.shape[1],1)), X_test.reshape((-1,daysago+1+dfstddev.shape[1],1)), Y_train.reshape((-1,daysforward)), Y_test.reshape((-1,daysforward))

    return X_train, X_test, Y_train, Y_test
print(TickerIntoTensor('aapl',200,1,0.9,20)[0].shape)