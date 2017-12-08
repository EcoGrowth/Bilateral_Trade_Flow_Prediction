#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:24:21 2017

@author: daikikumazawa
"""

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
from keras import backend as K
import matplotlib.pyplot as plt


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - (SS_res/(SS_tot + K.epsilon()))
    #return SS_res
    #return SS_tot

def retrieveData():
    data = pd.read_csv('data_2009_updated2.csv')

    print("Here is all our possible features!")
    print(data.columns.tolist())

    print ("We are picking ")
    features = ['GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord', 'Comlang',
                'Contig', 'OECD_o', 'OECD_d', 'GATT_d', 'GATT_o', 'XPTOT_o',]
    print(features)
    print ("Our dependent variable is 'FLOW'")

    # Data cleansing
    data = data[features + ['FLOW']]
    data = data.dropna(how='any')   # Dropping examples with NA features
    data = data[data.FLOW != 0]     # Dropping examples with FLOW = 0


    # Separate into train set & test set
    train, test = train_test_split(data, test_size=0.3)
    
    x_train = train[features]
    x_test = test[features]
    y_train = train['FLOW']
    y_test = test['FLOW']

    return x_train, x_test, y_train, y_test

def neuralNet(x_train, y_train, x_test, y_test):
    m, n = x_train.shape
    
    model = Sequential()
    model.add(Dense(10, activation = 'sigmoid', input_dim = n, 
                    kernel_regularizer = regularizers.l2(0.0001)))
    model.add(Dense(1, activation = 'linear'))

    # For some reason, stochastic gradient descent performs horribly
    model.compile(optimizer = 'rmsprop', loss='mse', metrics = [r2])
    
    model.fit(x_train, y_train, epochs = 40, batch_size = 32)
    score = model.test_on_batch(x_test, y_test)
    
    return score
    
def convertToLog(x, y, log_transform_list = ['GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord', 'XPTOT_o']):
    y = y.apply(np.log)
    for col in log_transform_list:
        x[col] = x[col].apply(np.log)

    return x, y    
    

def convertToNumpyArray(x_train, x_test, y_train, y_test):
    # Convert to np arrays 
    x_train = df.as_matrix(x_train)
    x_test = df.as_matrix(x_test)
    y_train = df.as_matrix(y_train)
    y_test = df.as_matrix(y_test)
    
    return x_train, x_test, y_train, y_test

def kFoldValidation(model, feature_data, result_data):
    print(model)
    scores = cross_val_score(model, feature_data, result_data, cv=10, verbose=1)
    print('Test Score')
    print(np.mean(scores))

x_train, x_test, y_train, y_test = retrieveData()

x_train, y_train = convertToLog(x_train, y_train)
x_test, y_test = convertToLog(x_test, y_test)

lr = LinearRegression()
#lr = Ridge(alpha=7.0)
#lr = KernelRidge(alpha=10^-2, kernel='rbf')
lr.fit(x_train, y_train)
print(lr.score(x_test, y_test))

x_train, x_test, y_train, y_test = convertToNumpyArray(x_train, x_test, y_train, y_test)
print(neuralNet(x_train, y_train, x_test, y_test))
