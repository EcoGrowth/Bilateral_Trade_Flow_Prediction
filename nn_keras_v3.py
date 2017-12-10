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

# Computes r^2 (coefficient of determination)
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - (SS_res/(SS_tot + K.epsilon()))

# Retrieves data in pandas dataframe format
def retrieveData():
    data = pd.read_csv('data_2009_updated2.csv')

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
    #print(sortByFLOW[:30])
    
    #print(data['FLOW'].min())

# Splits data into training set and test set  
def splitTrainTest(data, features):
    # Separate into train set & test set
    train, test = train_test_split(data, test_size=0.3)
    
    x_train = train[features]
    x_test = test[features]
    y_train = train['FLOW']
    y_test = test['FLOW']

    return x_train, x_test, y_train, y_test

# Runs a NN with a linear activation function in the output layer
def neuralNetRegression(x_train, y_train, x_test, y_test):
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

# Runs a NN with a softmax activation function in the output layer
def neuralNetSoftmax(x_train, y_train_disc, x_test, y_test_disc):
    m, n = x_train.shape
    
    model = Sequential()
    model.add(Dense(10, activation = 'sigmoid', input_dim = n, 
                    kernel_regularizer = regularizers.l2(0.0001)))
    model.add(Dense(10, activation = 'softmax'))

    # For some reason, stochastic gradient descent performs horribly
    model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit(x_train, y_train_disc, epochs = 40, batch_size = 32)
    score = model.test_on_batch(x_test, y_test_disc)
    
    return score
 
# Takes the log of features
def convertToLog(x, y, log_transform_list = ['GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord', 'XPTOT_o']):
    y = y.apply(np.log)
    for col in log_transform_list:
        x[col] = x[col].apply(np.log)

    return x, y    
    
# Discretize the output space using pandas qcut
def discretizeDataQCut(y):
    y_temp = pd.qcut(y, 10, labels = range(10))
    y_disc = np.zeros((y.shape[0], 10))
    y_disc[np.arange(y.shape[0]), y_temp] = 1
    
    print(y_disc.shape)
    return y_disc

# Discretize the output space using pandas cut
def discretizeDataCut(y):
    y_temp = pd.cut(y, 10, labels = range(10))
    y_disc = np.zeros((y.shape[0], 10))
    y_disc[np.arange(y.shape[0]), y_temp] = 1
    
    print(y_disc.shape)
    return y_disc

# Converts data in dataframe to numpy arrays
def convertToNumpyArray(x_train, x_test, y_train, y_test):
    # Convert to np arrays 
    x_train = df.as_matrix(x_train)
    x_test = df.as_matrix(x_test)
    y_train = df.as_matrix(y_train)
    y_test = df.as_matrix(y_test)
    
    return x_train, x_test, y_train, y_test

# K fold validation for linear regressions 
def kFoldValidation(model, feature_data, result_data):
    print(model)
    scores = cross_val_score(model, feature_data, result_data, cv=10, verbose=1)
    print('Test Score')
    print(np.mean(scores))


# Specify the features we want to use in our mo
#features = ['iso_o','iso_d','GDP_o', 'GDP_d' ,'year', 'POP_o', 'POP_d', 'Dist_coord', 'Comlang',
#                'Contig', 'OECD_o', 'OECD_d', 'GATT_d', 'GATT_o', 'XPTOT_o','EU_o',
#                'EU_d', 'Evercol']
features = ['GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord', 'Comlang',
                'Contig', 'OECD_o', 'OECD_d', 'GATT_d', 'GATT_o', 'XPTOT_o']

# Retrieve data
data = retrieveData()
# Clean data
data = dataCleansing(data, features)
# plotData(data)

x_train, x_test, y_train, y_test = splitTrainTest(data, features)
x_train, y_train = convertToLog(x_train, y_train)
x_test, y_test = convertToLog(x_test, y_test)

y_train_disc = discretizeDataQCut(y_train)
y_test_disc = discretizeDataQCut(y_test)

#y_train_disc = discretizeDataCut(y_train)
#y_test_disc = discretizeDataCut(y_test)

#
#lr = LinearRegression()
##lr = Ridge(alpha=7.0)
##lr = KernelRidge(alpha=10^-2, kernel='rbf')
#lr.fit(x_train, y_train)
#print(lr.score(x_test, y_test))
#
#x_train, x_test, y_train_disc, y_test_disc = convertToNumpyArray(x_train, x_test,
#                                                                 y_train_disc, y_test_disc)

print(neuralNetSoftmax(x_train, y_train_disc, x_test, y_test_disc))
