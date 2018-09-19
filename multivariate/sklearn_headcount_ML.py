# -*- coding: utf-8 -*-
"""
    Created on Mon Dec 08 09:44:33 2017
    
    @author: Suvarna.Krishnan
    """

import os
import pandas as pd
import numpy as np
import random

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import matplotlib.cm as cm

dir_path = os.path.dirname(os.path.realpath('__file__'))
print(dir_path)
e2015 = pd.read_excel(dir_path+"/Mockup_Extract_For_2015.xlsx")
e2016 = pd.read_excel(dir_path+"/Mockup_Extract_For_2016.xls")
e2017 = pd.read_excel(dir_path+"/Mockup_Extract_For_2017.xls")

headcounts = pd.concat([e2015,e2016,e2017], axis = 0)
headcounts['Store Id'] = headcounts['Store Id'].map(lambda x: str(x)[1:])
headcounts['Store Id'] = headcounts['Store Id'].apply(pd.to_numeric)

#Expand dateSkey
def expandDate(dateSkey):
    from datetime import datetime
    d = datetime.strptime(str(dateSkey), '%Y%m%d').date()
    return d

headcounts['dateSkey'] = headcounts['Week Start Date Skey'].apply(lambda x: expandDate(x))
headcounts['Year'] = headcounts['dateSkey'].apply(lambda x: x.year)
headcounts['Month'] = headcounts['dateSkey'].apply(lambda x: x.month)
#headcounts['Day'] = headcounts['dateSkey'].apply(lambda x: x.day)
headcounts['week'] = headcounts['dateSkey'].apply(lambda x: x.isocalendar()[1])

backupCopy1 = headcounts.copy()



#Removing all the stores which have a low count ( less than 22) 22 is the first quartile of unique occurences
storeCount = headcounts['Store Id'].value_counts()
#print(storeCount)
headcounts = headcounts.loc[(headcounts['Store Id'].isin(storeCount.index[storeCount >= 22]))]
headcounts = headcounts.reset_index(drop = True)
backupCopy = headcounts.copy()

headcounts = headcounts.drop(['Week Start Date Skey','dateSkey'], axis = 1)


X = headcounts.drop(['FT Count' , 'PT Count', 'Sched Effectiveness'], axis = 1)
Y = headcounts[['FT Count','PT Count']]

year = 2016
month = 9
storeNumber = 5
UniqueStore = X.drop_duplicates(['Store Id'])[['Store Id']].sample(storeNumber).reset_index(drop = True)
storeId = UniqueStore['Store Id'].tolist()
testX = X.loc[(X['Year'] == year) & (X['Month'] >= month) & (X['Store Id'].isin(storeId))]
testY = Y.iloc[testX.index,:]

trainX = X[~X.index.isin(testX.index)]
trainY = Y[Y.index.isin(trainX.index)]
dateIndex = backupCopy.loc[(backupCopy['Year'] == year) & (backupCopy['Month'] >= month) & (backupCopy['Store Id'].isin(storeId))]
#dateIndex = backupCopy.loc[testY.index,backupCopy.columns == 'dateSkey'].reset_index(drop = True)
dateIndex = dateIndex[['dateSkey']]
#dateIndex = pd.DataFrame(dateIndex)

#Fit Estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=300, max_features=len(X.columns),
                                       random_state=0),
                                       "K-nn": KNeighborsRegressor(n_neighbors = 7,algorithm = 'brute', weights = 'uniform'),
                                       "Linear regression": LinearRegression(),
                                       "RF regression": RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=18,
                                                                              max_features='auto', max_leaf_nodes=None,
                                                                              min_samples_leaf=1, min_samples_split=2,
                                                                              min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
                                                                              oob_score=False, random_state=0, verbose=0, warm_start=True)
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(trainX, trainY)
    y_test_predict[name] = estimator.predict(testX)

fullTime = testY['FT Count']
fullTime = fullTime.reset_index()
partTime = testY['PT Count']
partTime = partTime.reset_index()

for key in y_test_predict:
    value = y_test_predict[key]
    value = pd.DataFrame(value)
    ft = pd.DataFrame({key:value[value.columns[0]]})
    ft = ft.astype(int)
    pt = pd.DataFrame({key:value[value.columns[1]]})
    pt = pt.astype(int)
    fullTime = pd.concat([fullTime, ft], axis = 1)
    partTime = pd.concat([partTime, pt], axis = 1)



for i in (testX['Store Id'].unique()):
    print(testX['Store Id'].unique())
    indexStore = testX.index.values[testX['Store Id'] == i]
    subsetDataFT = fullTime.loc[fullTime['index'].isin(indexStore)]
    subsetDataFT.index = dateIndex.iloc[subsetDataFT.index,0]
    subsetDataPT = partTime.loc[partTime['index'].isin(indexStore)]
    subsetDataPT.index = dateIndex.iloc[subsetDataPT.index,0]
    
    for j in range(2, len(subsetDataFT.columns)):
        print(subsetDataFT.columns)
        temp1 = subsetDataFT.iloc[:,[1,j]]
        #temp1.plot(style = ['x','.'],grid = 1, title = i)
        temp2 = subsetDataPT.iloc[:,[1,j]]
        temp = pd.concat([temp1,temp2], axis = 1)
        temp.plot(style = ['-','o','-','o'], grid = 0.2, title = i)        
        plt.show()
        plt.close()


x = headcounts['Store Id'].value_counts()