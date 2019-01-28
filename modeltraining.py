# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:02:09 2018

@author: Jun Wei
"""

from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

xlsx = pd.ExcelFile('tempdat.xlsx')

def preprocess(df):
    time = df.values[:9,0].reshape(-1,1); temp = df.values[:9,1].reshape(-1,1)
    delta_t = temp - temp[0][0]
    poly_time = PolynomialFeatures(2,include_bias=False).fit_transform(time)
    maclaurin = linear_model.LinearRegression()
    maclaurin.fit(poly_time,delta_t)
    init_grad = maclaurin.coef_[0][0]; init_temp = temp[0][0]
    return [init_grad, init_temp]

class tempmodel:
    def __init__(self):
        self.model = linear_model.LinearRegression()
        
    def train(self,feat_list,target_list):
        grad_list = feat_list[:,0].reshape(-1,1)
        target_list -= feat_list[:,1].reshape(-1,1)
        self.model.fit(grad_list,target_list)
        return self.model.coef_
        
    def predict(self,features):
        init_grad = features[0]; init_temp = features[1]
        delta_t_final = self.model.predict(init_grad) + init_temp
        return delta_t_final[0][0]

framekeys = ['set1','set2','set3','set4','set5','set6','set7','set8','set9','set10','set11','set12']

features = []; final_temp = []
for n in range(1,13):
    dataframe = pd.read_excel(xlsx,sheetname = 'set'+str(n))
    features.append(preprocess(dataframe))
    ftemp = dataframe.values[0,3]
    final_temp.append(ftemp)
    
npfeat = np.asarray(features); npfintemp = np.asarray(final_temp)


#AN GUO: so the target variable is final temperature, the features are the initial rate of change of temperature and the initial temperature
#def display_scatter(x,y,xlabel='x',ylabel='y',title_name='default'):
#    plt.figure()
#    plt.scatter(x,y)
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    plt.suptitle(title_name)
#    plt.show()
#    
#display_scatter(npfeat[:,0],(npfintemp-npfeat[:,1]),'Initial Gradient','Final Temperature Change','Feature Scatter Plot')
model = tempmodel()

n = 1; seeds = [-1,44,36,419,5354,666,2048,777,555,12,1,69,0,13,343,2401,102800,441,999,17,39]; accuracy_list = []
while n  <= 20:
    print('Test ',n)
    feat_train, feat_test, fintemp_train, fintemp_test = train_test_split(npfeat,npfintemp,test_size=0.1,random_state = seeds[n])
    coef = model.train(feat_train,fintemp_train.reshape(-1,1))
    predtemp1 = model.predict(feat_test[0,:]); truetemp1 = fintemp_test[0]
    predtemp2 = model.predict(feat_test[1,:]); truetemp2 = fintemp_test[1]
    print('(predicted,actual) = ('+str(predtemp1)+','+str(truetemp1)+')')
    accuracy = 100 - abs(truetemp1-predtemp1)/truetemp1*100
    accuracy_list.append(accuracy)
    print('accuracy: ',accuracy)
    print('(predicted,actual) = ('+str(predtemp2)+','+str(truetemp2)+')')
    accuracy = 100 - abs(truetemp2-predtemp2)/truetemp1*100
    print('accuracy: ',accuracy)
    accuracy_list.append(accuracy)
    n += 1
    
avg_accuracy = sum(accuracy_list)/40
print('average accuracy: ',avg_accuracy)
    
model.train(npfeat,npfintemp.reshape(-1,1))
pickle.dump(model,open('model.p','wb'))
