#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:35:06 2022

@author: abdul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


plot_path = './plots/'
model_path = './models/'
data_path = './my_data/'

sales = pd.read_csv('sales_small.csv')

#sales = pd.read_csv('New_Sales_Data.csv')
sales = sales.rename(columns={'sales_small.ProductID': 'ProductID'})

time = sales['sales_small.WeekKey']

new_week = []
r = 52
for i in range(0,len(time)):
    if time[i] < 201600:
        x = int(str(time[i])[4:])
        new_week.append(x)
    else:
        x = int(str(time[i])[4:])+r
        new_week.append(x)

sales['week'] = new_week 


products = pd.read_csv('products.csv')
products = products.rename(columns={'products.ProductID': 'ProductID'})

merged = pd.merge(sales, products, on='ProductID')

x = merged[['sales_small.Channel',
 'sales_small.Country',
 'sales_small.CSP',
 'sales_small.StoreStockVolume',
 'sales_small.DepotStockVolume',
 'sales_small.FutureCommitmentVolume',
 'sales_small.IntakeVolume',
 'sales_small.StoresWithStockCount',
 'week',
 'products.Season',
 'products.Group',
 'products.SubGroup',
 'products.Class',
 'products.SubClass']]



y = merged['sales_small.SalesVolume']

merged['products.Group'].unique()

# Handling categorical data
#SPLITTING NUMERIC AND CATERGORIC FEATURES
n = 100

categorical_var = [var for var in x.columns if x[var].nunique() <=n]
numerical_var = [var for var in x.columns if x[var].nunique() >n]

categorical_var.remove('week')
numerical_var.append('week')

categoric_data = x[categorical_var]
numeric_data = x[numerical_var]


#VISAULAZING EACH CATEGORICAL FEATURE DISTRIBUTION

categoric_data['products.Group'].value_counts().plot.bar(figsize=(6, 4), rot=30)
plt.axhline(y=0.05*len(y), color='r', linestyle='-')
plt.suptitle('Group wise distribution', fontsize=12)
plt.ylabel('counts', fontsize=10)
plt.savefig(plot_path+'1.png')
plt.show()


categoric_data['products.SubGroup'].value_counts().plot.bar(figsize=(6, 4), rot=30)
plt.axhline(y=0.05*len(y), color='r', linestyle='-')
plt.suptitle('SubGroup wise distribution', fontsize=12)
plt.ylabel('counts', fontsize=10)
plt.savefig(plot_path+'2.png')
plt.show()



def replace_rare_categories(categoric_data):
    
    categorical_var = list(categoric_data)
    transformed_categoric = pd.DataFrame()
    rare = []
    for j in range(0,len(categorical_var)):
        feat = categoric_data[categorical_var[j]]
        b = feat.value_counts()/len(categoric_data)*100
        filt = b<5
        rare_cats = list(filt[filt].index)
        val =['rare']*len(rare_cats)
        dictionary = dict(zip(rare_cats, val))
        feat1 = feat.replace(dictionary) 
        transformed_categoric[categorical_var[j]]=feat1
        rare.append(rare_cats)
    return transformed_categoric, rare

transformed_categoric_data, rare_categories = replace_rare_categories(categoric_data)

# rechecking 

transformed_categoric_data['products.Group'].value_counts().plot.bar(figsize=(6, 4), rot=30)
plt.axhline(y=0.05*len(y), color='r', linestyle='-')
plt.suptitle('Group Rare Categories Summarization', fontsize=12)
plt.ylabel('counts', fontsize=10)
plt.savefig(plot_path+'3.png')
plt.show()


transformed_categoric_data['products.SubGroup'].value_counts().plot.bar(figsize=(10, 4), rot=30)
plt.axhline(y=0.05*len(y), color='r', linestyle='-')
plt.suptitle('SubGroup wise distribution', fontsize=12)
plt.ylabel('counts', fontsize=10)
plt.savefig(plot_path+'4.png')
plt.show()


# MAPPINGS
d1 = {'Stores': 0, 'Online': 1}
d2 = {'A': 0, 'B':1}
d3 = {'L':0, 'rare':1}
d4 = {'26387251':0, 'bca94c97':1, 'edf80f3a':2, '606565a1':3, 'rare':4}
d5 = {'c3567a18':0, '44b005af':1, '14edd834':2, 'edf80f3a':3, '33de6bfe':4, 
      'rare':5, 'c830b21c':6}
d6 = {'rare':0, '96d57a3c':1, 'bb900370':2, 'c7b04e18':3, '1ed9f279':4, 
      '6ad22ddd':5, 'bf003871':6}
d7 = {'rare':0, 'ef307285':1, '3906b741':2, '6797d7af':3}

x1 = transformed_categoric_data.copy()

x1= x1.replace({"sales_small.Channel": d1})
x1= x1.replace({"sales_small.Country": d2})
x1= x1.replace({"products.Season": d3})
x1= x1.replace({"products.Group": d4})
x1= x1.replace({"products.SubGroup": d5})
x1= x1.replace({"products.Class": d6})
x1= x1.replace({"products.SubClass": d7})


x2 = pd.concat([numeric_data, x1], axis=1)



summary = x2.describe()

#correlation
corr_matrix = x2.corr()
corr_matrix.style.background_gradient(cmap='coolwarm')
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
drop_feature = [column for column in upper.columns if any(upper[column] > 0.85)]

X_train, X_test, y_train, y_test = train_test_split(x2, y, test_size=0.3, random_state=41)


#SAVING TRANSFORMED TRIAN DATA
X_train.to_csv(data_path+'x_train.csv', index=False)
y_train.to_csv(data_path+'y_train.csv', index=False)

X_test.to_csv(data_path+'x_test.csv', index=False)
y_test.to_csv(data_path+'y_test.csv', index=False)


# your test data
#x2.to_csv(data_path+'x_test.csv', index=False)
#y.to_csv(data_path+'y_test.csv', index=False)

