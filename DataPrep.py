import numpy as np
import pandas as pd
import matplotlib as plt
import yfinance as yf
import keras 
import os
import datetime 
from sklearn.preprocessing import MinMaxScaler

def TickerIntoTensor(ticker='aapl',daysago=100,daysforward=1,split_percentage=0.9):
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
    dfstddev = pd.DataFrame()
    dfstddev['daysago# std'] = df.iloc[:,:daysago+1].std(axis=1)
    dfstddev['half of daysago# std'] = df.iloc[:,daysago//2:daysago+1].std(axis=1)
    dfstddev['third of daysago# std'] = df.iloc[:,daysago-(daysago//3):daysago+1].std(axis=1)
    dfstddev['quarter of daysago# std'] = df.iloc[:,daysago-(daysago//4):daysago+1].std(axis=1)
    dfstddev['ten days from daysago# std'] = df.iloc[:,daysago-10:daysago+1].std(axis=1)
    sc = MinMaxScaler(feature_range=(0,1))
    stddev_scaled = sc.fit_transform(dfstddev)
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
print(TickerIntoTensor('aapl',100,1,0.9)[0].shape)
