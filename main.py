#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:09:29 2022

@author: abdul
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from scipy.stats import pearsonr, chi2_contingency
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import seaborn as sns
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import *


plot_path = './plots/'
model_path = './models/'
data_path = './my_data/'


if __name__ == '__main__':

    x_test = pd.read_csv(data_path+'x_test.csv')
    y_test = pd.read_csv(data_path+'y_test.csv')
    
    y_test = np.log(y_test+1)
    y_test = pd.Series(y_test['sales_small.SalesVolume'])
    
    file_name = "xgb_reg.pkl"
    
    xgb_model_loaded = pickle.load(open(model_path+file_name, "rb"))
    
    yhat = xgb_model_loaded.predict(x_test)
    
    
    xgb_model_loaded.feature_importances_
    plt.barh(list(x_test), xgb_model_loaded.feature_importances_)
    
    
    
    # calculate Pearson's correlation
    corr, _ = pearsonr(y_test, yhat)
    corr = round(corr, 2)
    print('Pearsons correlation: %.3f' % corr)
    rmse= np.sqrt(np.square(y_test-yhat))
    print('Avg RMSE:', np.average(rmse))
    rmse = round(np.average(rmse), 2)
    
    text0 = "Corr:"+str(corr)+"  RMSE:"+str(rmse)
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.scatter(y_test, yhat)
    plt.title('XGBoost Predictions on test set: transformed log(y)')
    p1, p2 = [0, max(y_test)], [0, max(y_test)]
    plt.plot(p1, p2, color ='red') 
    plt.text(0, 6, text0, fontsize=12)
    plt.xlabel('log(y_test)')
    plt.ylabel('log(pred)')
    plt.savefig(plot_path+'7.png')
    plt.show()
    
    
    y1 = np.exp(y_test)-1
    y2 = np.exp(yhat)-1
    
    corr, _ = pearsonr(y1, y2)
    corr = round(corr, 2)
    print('Pearsons correlation: %.3f' % corr)
    rmse= np.sqrt(np.square(y1- y2))
    print('Avg RMSE:', np.average(rmse))
    rmse = round(np.average(rmse), 2)
    
    text0 = "Corr:"+str(corr)+"  RMSE:"+str(rmse)
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.scatter(y1, y2)
    plt.title('XGBoost Predictions: y')
    p1, p2 = [0, max(y1)], [0, max(y1)]
    plt.plot(p1, p2, color ='red') 
    plt.text(0, 750, text0, fontsize=12)
    plt.xlabel('y_test')
    plt.ylabel('pred')
    plt.savefig(plot_path+'8.png')
    plt.show()
    
    
    #============Neural Network
    
    model = load_model(model_path+'model.h5')
    
    # data normalization with sklearn
    # transform testing dataabs
    norm = pickle.load(open(data_path+'norm.pkl', "rb"))
    x_test = norm.transform(x_test)
    
    
    y_pred = model.predict(x_test)
    pred1 = [item for sublist in y_pred for item in sublist]
    pred2= pd.Series(pred1)
    
    # calculate Pearson's correlation
    corr, _ = pearsonr(y_test, pred2)
    corr = round(corr, 2)
    print('Pearsons correlation: %.3f' % corr)
    rmse= np.sqrt(np.square(y_test- pred2))
    print('Avg RMSE:', np.average(rmse))
    rmse = round(np.average(rmse), 2)
    
    text0 = "Corr:"+str(corr)+"  RMSE:"+str(rmse)
    
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.scatter(y_test, pred2)
    plt.title('Neural Network Predictions')
    p1, p2 = [0, max(y_test)], [0, max(y_test)]
    plt.plot(p1, p2, color ='red')
    plt.text(0, 6, text0, fontsize=12)
    plt.xlabel('log(y_test)')
    plt.ylabel('log(pred)')
    plt.savefig(plot_path+'10.png')
    plt.show()
    
    
    
    y1 = np.exp(y_test)-1
    y22 = np.exp(pred2)-1
    
    corr, _ = pearsonr(y1, y22)
    corr = round(corr, 2)
    print('Pearsons correlation: %.2f' % corr)
    rmse= np.sqrt(np.square(y1- y22))
    print('Avg RMSE:', np.average(rmse))
    rmse = round(np.average(rmse), 2)
    
    text1 = "Corr:"+str(corr)+"  RMSE:"+str(rmse)
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.scatter(y1, y22)
    plt.title('Neural Network Predictions')
    p1, p2 = [0, max(y1)], [0, max(y1)]
    plt.plot(p1, p2, color ='red')
    plt.text(0, 750, text1, fontsize=12)
    plt.xlabel('y_test')
    plt.ylabel('pred')
    plt.savefig(plot_path+'11.png')
    plt.show()
    
    
    # Predictions
    my_predictions = pd.DataFrame(columns = ['xgboost','Neural Network'])
    my_predictions['xgboost'] = pd.Series(y2)
    my_predictions['Neural Network'] = y22
    
    
    my_predictions.to_csv(data_path+'my_predictions.csv', index=False)





