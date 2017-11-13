import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np


def retrieveData():
    data = pd.read_csv('data_2009_updated2.csv')

    print("Here is all our possible features!")
    print data.columns.tolist()

    print ("We are picking ")
    features = ['GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord', 'Comlang', 'Contig', 'OECD_o', 'OECD_d', 'GATT_d', 'GATT_o', 'XPTOT_o', 'XPTOT_d']
    print features
    print ("Our dependent variable is 'FLOW")

    # clean data
    data = data[features + ['FLOW']]
    data = data.dropna(how='any')
    data = data[data.FLOW != 0]

    wwdata = data.tail(10000)
    data = data.sample(10000)
    #print data.shape

    features = data[features]
    dependent_variable = data['FLOW']
    return features, dependent_variable




def kFoldValidation(model, feature_data, result_data):
    print model
    scores = cross_val_score(model, feature_data, result_data, cv=10, verbose=1)
    print 'Test Score'
    print np.mean(scores)



def convertToGravityModel(feature_data, result_data, log_transform_list = ['GDP_o', 'GDP_d', 'POP_o', 'POP_d', 'Dist_coord']):
    result_data = result_data.apply(np.log)
    for col in log_transform_list:
        feature_data[col] = feature_data[col].apply(np.log)

    return feature_data, result_data



feature_data, result_data = retrieveData()

feature_data, result_data = convertToGravityModel(feature_data, result_data)

lr = LinearRegression()
#lr = Ridge(alpha=7.0)
#lr = KernelRidge(alpha=10^-2, kernel='rbf')
kFoldValidation(lr, feature_data, result_data)

