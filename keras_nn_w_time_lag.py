#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:24:21 2017

@author: daikikumazawa
"""
import math
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from keras.models import Sequential
import keras.regularizers as regularizers
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Computes r^2 (coefficient of determination)
# Reference: http://jmbeaujour.com/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - (SS_res/(SS_tot + K.epsilon()))

# Retrieves data in pandas dataframe format
def retrieveData():
    data = pd.read_csv('tradhist_data_with_time_lag.csv')

    print("Here is all our possible features!")
    print(data.columns.tolist())
    
    return data

# Cleans the data
def dataCleansing(data, features):
    print ("We are picking ")
    print(features)
    print ("Our dependent variable is 'FLOW'")

    # Data cleansing
    data = data[features + ['FLOW']]  # Dropping irrelevant features
    data = data.dropna(how='any')   # Dropping examples with NA features
    data = data[data.FLOW >= 100]     # Dropping examples with FLOW < 100
    data = data[data.FLOW_lag >= 100]
    
    return data

# Plots the features & output
def plotData(data):
    
    # sortByFLOW = data.sort_values('FLOW')
    # print(data['FLOW'].max())
    plt.title('FLOW')
    plt.hist(np.log(data['FLOW']), bins = 50)
    plt.show()
    plt.title('GDP_o')
    plt.hist(np.log(data['GDP_o']), bins = 50)
    plt.show()
    plt.title('POP_o')
    plt.hist(np.log(data['POP_o']), bins = 50)
    plt.show()
    plt.title('Dist_coord')
    plt.hist(np.log(data['Dist_coord']), bins = 50)
    plt.show()
    plt.title('XPTOT')
    plt.hist(np.log(data['XPTOT_o']), bins = 50)
    plt.show()


# Splits data into training set and test set  
def splitTrainTest(data, features):
    # Separate into train set & test set
    train, test = train_test_split(data, test_size=0.3)
    
    x_train = train[features]
    x_test = test[features]
    y_train = train['FLOW']
    y_test = test['FLOW']

    return x_train, x_test, y_train, y_test


def neuralNet(x_train, y_train, x_test, y_test, activation='selu', epochs=40, optimizer='adam', width=48, depth=3):
    m, n = x_train.shape

    model = Sequential()
    model.add(Dense(width, activation=activation, input_dim=n,
                    kernel_regularizer=regularizers.l2(0.0001)))
    for i in range(depth - 1):
        model.add(Dense(width, activation=activation,
                        kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dense(width, activation=activation,
                        kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(1, activation='linear'))

    # For some reason, stochastic gradient descent performs horribly
    model.compile(optimizer=optimizer, loss='mse', metrics=[r2])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=1000, verbose=2)
    score = model.test_on_batch(x_test, y_test)
    return score, history, model

# Takes the log of features
def convertToLog(x, y, log_transform_list = ['FLOW_lag', 'GDP_o', 'GDP_d', 'POP_o', 
                                             'POP_d', 'Dist_coord', 'XPTOT_o']):
    y = y.apply(np.log)
    for col in log_transform_list:
        x[col] = x[col].apply(np.log)

    return x, y    
    

# Specify the features we want to use in our mo
#features = ['iso_o','iso_d','GDP_o', 'GDP_d' ,'year', 'POP_o', 'POP_d', 'Dist_coord', 'Comlang',
#                'Contig', 'OECD_o', 'OECD_d', 'GATT_d', 'GATT_o', 'XPTOT_o','EU_o',
#                'EU_d', 'Evercol']
features = ['FLOW_lag', 'GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord', 'Comlang',
             'Contig', 'OECD_o', 'OECD_d', 'GATT_d', 'GATT_o', 'XPTOT_o']
#features = ['FLOW_lag']
# features2 = ['GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord', 'Comlang',
#                'Contig', 'OECD_o', 'OECD_d', 'GATT_d', 'GATT_o', 'XPTOT_o',]

# Retrieve data
data = retrieveData()
# Clean data
data = dataCleansing(data, features)
data = data[['FLOW', 'FLOW_lag']]
# plotData(data)

x_train, x_test, y_train, y_test = splitTrainTest(data, ['FLOW_lag'])
x_train, y_train = convertToLog(x_train, y_train, log_transform_list=['FLOW_lag'])
x_test, y_test = convertToLog(x_test, y_test, log_transform_list=['FLOW_lag'])

# x_train = preprocessing.scale(x_train)
# x_test = preprocessing.scale(x_test)


#y_train_disc = discretizeDataQCut(y_train)
#y_test_disc = discretizeDataQCut(y_test)
#
##y_train_disc = discretizeDataCut(y_train)
##y_test_disc = discretizeDataCut(y_test)
#lr = LinearRegression()
#
#lr.fit(x_train[features2], y_train)
#y_pred = lr.predict(x_test[features2])
#print(math.sqrt(mean_squared_error(y_test, y_pred)))
#print(lr.score(x_test[features2], y_test))
#
#
lr = LinearRegression()

lr.fit(x_train, y_train)
y_test_pred = lr.predict(x_test)
y_train_pred = lr.predict(x_train)

print(math.sqrt(mean_squared_error(y_train, y_train_pred)))
print(lr.score(x_train, y_train))
print(math.sqrt(mean_squared_error(y_test, y_test_pred)))
print(lr.score(x_test, y_test))
#
#x_train = x_train['FLOW_lag']
#x_test = x_test['FLOW_lag']
#
##
#lr = LinearRegression()
###lr = Ridge(alpha=7.0)
###lr = KernelRidge(alpha=10^-2, kernel='rbf')
#lr.fit(x_train.reshape(-1,1), y_train)
#y_pred = lr.predict(x_test.reshape(-1,1))
#print(math.sqrt(mean_squared_error(y_test, y_pred)))
#print(lr.score(x_test.reshape(-1,1), y_test))
##


#print(neuralNet(x_train, y_train, x_test, y_test, epochs=500))
