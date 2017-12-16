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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import preprocessing
from util import convertToLog

FEATURES= ['GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord', 'Comlang',
                'Contig', 'OECD_o', 'OECD_d', 'GATT_d', 'GATT_o', 'XPTOT_o']


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - (SS_res/(SS_tot + K.epsilon()))
    #return SS_res
    #return SS_tot

def retrieveData():
    data = pd.read_csv('tradehist_data.csv')

    print("Here is all our possible features!")
    print(data.columns.tolist())

    print ("We are picking ")
    countries = ['iso_o', 'iso_d']
    print(FEATURES)
    print ("Our dependent variable is 'FLOW'")

    # Data cleansing
    data = data[FEATURES+ ['FLOW'] + countries + ['year']]
    data = data.dropna(how='any')   # Dropping examples with NA features
    data = data[data.FLOW >= 100]  # Dropping examples with FLOW = 0



    # Separate into train set & test set
    train, test = train_test_split(data, test_size=0.3)
    
    x_train = train[FEATURES]
    y_train = train['FLOW']
    testLux = None

    #test = test.groupby('iso_o')
    selectMaxCountry = False
    if selectMaxCountry:
        test = test[test['iso_o'] == 'NGA']
        test = test[test['year'] == 2012]

    x_test = test[FEATURES]
    y_test = test['FLOW']

    return x_train, x_test, y_train, y_test, test, test.groupby('year')

def neuralNet(x_train, y_train, x_test, y_test, activation = 'selu', epochs = 40, optimizer='adam', width = 48, depth = 3):
    m, n = x_train.shape
    
    model = Sequential()
    model.add(Dense(width, activation = activation, input_dim = n,
                    kernel_regularizer = regularizers.l2(0.0001)))
    for i in range(depth-1):
        model.add(Dense(width, activation = activation,
                    kernel_regularizer = regularizers.l2(0.0001)))
        model.add(Dense(width, activation = activation,
                    kernel_regularizer = regularizers.l2(0.0001)))
    model.add(Dense(1, activation = 'linear'))

    # For some reason, stochastic gradient descent performs horribly
    model.compile(optimizer = optimizer, loss='mse', metrics = [r2])
    
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = 1000, verbose=2)
    score = model.test_on_batch(x_test, y_test)
    return score, history, model


def kFoldValidation(model, feature_data, result_data):
    print(model)
    scores = cross_val_score(model, feature_data, result_data, cv=10, verbose=2)
    print('Test Score')
    print(np.mean(scores))

x_train, x_test, y_train, y_test, test_w, test_g = retrieveData()


x_train, y_train = convertToLog(x_train, y_train)
x_test, y_test = convertToLog(x_test, y_test)

x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)


lr = LinearRegression()
#lr = Ridge(alpha=7)
#lr = KernelRidge(alpha=.5, kernel='rbf', degree=2)
#kFoldValidation(lr, x_train, y_train)
#
lr.fit(x_train, y_train)
print lr.coef_
print mean_squared_error(y_train, lr.predict(x_train))
print(lr.score(x_train, y_train))
print "HI"
print mean_squared_error(y_test, lr.predict(x_test))
print(lr.score(x_test, y_test))


epochs = 3000
score8, hist8, model = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='adam', width=48, depth=3)
print score8

#
# lux_predict_g = lr.predict(x_test)
# lux_predict = model.predict(x_test)
# lux_actual = y_test
# lux_countries = test_w['iso_d']
# plot_size = 30
#
# xticks = lux_countries[:plot_size]
# x = np.arange(lux_countries.shape[0])[:plot_size]
# plt.plot(x,lux_predict[:plot_size], label="Model 4")
# plt.plot(x, lux_actual[:plot_size], label="Actual")
# plt.plot(x, lux_predict_g[:plot_size], label="Model 2")
# plt.xticks(x,xticks)
# plt.legend()
# plt.title("Trade Flow Per Country With Source Country Nigeria, 2012")
# plt.show()


# temp = []
# #
# for name, group in test_g:
#     print name
#     print group.shape
#     group_y = group['FLOW']
#     group_x = group[FEATURES]
#     group_x, group_y = convertToLog(group_x, group_y)
#     group_x = preprocessing.scale(group_x)
#     networkScore = model.test_on_batch(group_x, group_y)[1]
#     if networkScore > 0:
#         temp.append((name, networkScore - lr.score(group_x, group_y), networkScore))
# #
# print sorted(temp, key=lambda tup: tup[1], reverse=True)
#
#
# testLux_x = testLux[FEATURES]
# testLux_y = testLux['FLOW']
# testLux_x, testLux_y = convertToLog(testLux_x, testLux_y)
# testLux_x = preprocessing.scale(testLux_x)
# print model.test_on_batch(testLux_x, testLux_y)
# print lr.score(testLux_x, testLux_y)



epochs = 3000
#score1, hist1 = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='adam', width=6, depth=1)
#score2, hist2 = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='adam', width=6, depth=3)
#score3, hist3 = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='adam', width=6, depth=5)
#score4, hist4 = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='adam', width=12, depth=1)
#score5, hist5 = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='adam', width=12, depth=3)
#score6, hist6 = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='adam', width=12, depth=5)
#score7, hist7 = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='adam', width=48, depth=1)

#score9, hist9 = neuralNet(x_train, y_train, x_test, y_test, activation='tanh', epochs=epochs, optimizer='rmsprop', width=48, depth=5)

#loss1 = hist1.history['loss']
# loss2 = hist2.history['loss']
# loss3 = hist3.history['loss']
# loss4 = hist4.history['loss']
# print score1
# print score2
# print score3
# print score4
# print score5
# print score6
# print score7
# print score8
#print score9
#
# plt.plot(np.arange(start= 50, stop=epochs), loss1[50:], label="SGD")
# plt.plot(np.arange(start = 50, stop = epochs), loss2[50:], label="ADAM")
# plt.plot(np.arange(start = 50, stop = epochs), loss3[50:], label="ADAGRAD")
# plt.plot(np.arange(start = 50, stop = epochs), loss4[50:], label="RMSPROP")
# plt.ylabel("Mean Square Error")
# plt.xlabel("Epoch")
# plt.legend()
# plt.title("Tanh Activation using different Optimizers")
# plt.show()