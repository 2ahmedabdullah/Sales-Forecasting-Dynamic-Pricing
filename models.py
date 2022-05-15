#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:42:15 2022

@author: abdul
"""


import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from scipy.stats import pearsonr, chi2_contingency
from sklearn import metrics
import seaborn as sns
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


plot_path = './plots/'
model_path = './models/'
data_path = './my_data/'

x_train = pd.read_csv(data_path+'x_train.csv')
y_train = pd.read_csv(data_path+'y_train.csv')

x_test = pd.read_csv(data_path+'x_test.csv')
y_test = pd.read_csv(data_path+'y_test.csv')

y_test = np.log(y_test+1)
y_train = np.log(y_train+1)


#=======================XGBOOST=====================================================


xgb = XGBRegressor()
xgb.fit(x_train, y_train)
yhat = xgb.predict(x_test)
y_test = pd.Series(y_test['sales_small.SalesVolume'])
#yhat = pd.Series(yhat)

file_name = "xgb_reg.pkl"

# save
#pickle.dump(xgb, open(model_path+file_name, "wb"))
#xgb_model_loaded = pickle.load(open(model_path+file_name, "rb"))


xgb.feature_importances_
plt.barh(list(x_train), xgb.feature_importances_)

#Available importance_types = [‘weight’, ‘gain’, ‘cover’, total_gain, total_cover]

feature_imp = xgb.get_booster().get_score(importance_type = 'gain')
key = list(feature_imp.keys())
value = list(feature_imp.values())
top_feats = pd.DataFrame({'Metric': key, 'Importance': value})
top_feats = top_feats.sort_values('Importance', ascending = False)
top_feats['Importance'] = top_feats['Importance']/top_feats['Importance'].sum()*100
top_feats = top_feats.set_index(top_feats['Metric'])

'''
#plot
top_feats.plot(kind = 'barh', title = 'Most Important Features', color = 'blue').invert_yaxis()
plt.axvline(x = 5, ymin = 0, ymax= 10, color ='red')
plt.show()
'''


#shapley plots
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(x_train)

shap.summary_plot(shap_values, features = x_train, feature_names = x_train.columns,
                  show = False, plot_size = (10, 7), max_display = len(x_train.columns))
plt.title('Shap Plot', fontsize = 20)
plt.show()



#importance acc to shapley values
mean_shaps = np.abs(shap_values).mean(0)
shap_importance = pd.DataFrame(list(zip(x_train.columns, mean_shaps)), columns = ['Metric', 'Mean Shap Value'])
shap_importance['Mean Shap Value'] = shap_importance['Mean Shap Value']/shap_importance['Mean Shap Value'].sum()*100
shap_importance = shap_importance.sort_values('Mean Shap Value', ascending = False)
shap_importance = shap_importance.set_index(shap_importance['Metric'])

shap_importance.plot(kind = 'barh', title = 'Most Important Features', color = 'blue').invert_yaxis()
plt.axvline(x = 5, ymin = 0, ymax= 10, color ='red')
plt.xlabel('% Importance')
z = shap_importance['Mean Shap Value'].to_list()
m = [i for i in z if i >= 5]
plt.xlim(0, max(z)*1.2)
for i in range(0, len(m)):
    plt.text(z[i]+0.5, i, str(int(round(z[i], 0))) + '%', color='black', fontweight='bold', fontsize=10, va='center')
plt.show()


plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(y_test, yhat)
plt.title('XGBoost Predictions on test set: transformed log(y)')
p1, p2 = [0, max(y_test)], [0, max(y_test)]
plt.plot(p1, p2, color ='red') 
plt.xlabel('log(y_test)')
plt.ylabel('log(pred)')
plt.savefig(plot_path+'7.png')
plt.show()



# calculate Pearson's correlation
corr, _ = pearsonr(y_test, yhat)
print('Pearsons correlation: %.3f' % corr)
rmse= np.sqrt(np.square(y_test-yhat))
print('Avg RMSE:', np.average(rmse))


y1 = np.exp(y_test)-1
y2 = np.exp(yhat)-1

r2 = r2_score(y1, y2)
r2 =round(r2,2)

corr, _ = pearsonr(y1, y2)
corr = round(corr, 2)
print('Pearsons correlation: %.3f' % corr)
rmse= np.sqrt(np.square(y1- y2))
print('Avg RMSE:', np.average(rmse))
rmse = round(np.average(rmse), 2)

text0 = "R2:"+str(r2)+"  RMSE:"+str(rmse)

plt.figure(figsize=(5, 3), dpi=80)
plt.scatter(y1, y2)
plt.title('XGBoost Predictions')
p1, p2 = [0, max(y1)], [0, max(y1)]
plt.plot(p1, p2, color ='red') 
plt.text(0, 750, text0, fontsize=12)
plt.xlabel('y_test')
plt.ylabel('pred')
plt.savefig(plot_path+'8.png')
plt.show()


#=======================Neural Network==============================================



# data normalization with sklearn
# fit scaler on training data
norm = MinMaxScaler().fit(x_train)
#pickle.dump(norm, open(data_path+'norm.pkl', "wb"))

# transform training data
x_train = norm.transform(x_train)

# transform testing dataabs
x_test = norm.transform(x_test)

act = 'softplus'

#MODEL BUILDING
input_dim = 14

model = Sequential()
model.add(Dense(14, activation=act, input_shape=(input_dim,)))
model.add(Dense(10, activation=act))
model.add(Dense(7, activation=act))
model.add(Dense(3, activation=act))
model.add(Dense(1, activation=act))
model.compile(loss='mse', optimizer = Adam())

model1= model.fit(x_train, y_train, batch_size=64, epochs=500, 
                                verbose=1, validation_data=(x_test, y_test))

#model.save(model_path+'model.h5')

plt.figure(figsize=(4, 4), dpi=80)
plt.plot(model1.history['loss'])
plt.plot(model1.history['val_loss'])
plt.title('Neural Network loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
#plt.savefig(plot_path+'9.png')    
plt.show()


y_pred = model.predict(x_test)
pred1 = [item for sublist in y_pred for item in sublist]
pred2= np.array(pred1)


plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(y_test, pred2)
plt.title('Neural Network Predictions')
p1, p2 = [0, max(y_test)], [0, max(y_test)]
plt.plot(p1, p2, color ='red') 
plt.xlabel('log(y_test)')
plt.ylabel('log(pred)')
#plt.savefig(plot_path+'10.png')
plt.show()




# calculate Pearson's correlation
corr, _ = pearsonr(y_test, pred2)
print('Pearsons correlation: %.3f' % corr)
rmse= np.sqrt(np.square(y_test- pred2))
print('Avg RMSE:', np.average(rmse))


y1 = np.exp(y_test)-1
y2 = np.exp(pred2)-1

r2 = r2_score(y1, y2)
r2 = round(r2, 2)

corr, _ = pearsonr(y1, y2)
corr = round(corr, 2)
print('Pearsons correlation: %.2f' % corr)
rmse= np.sqrt(np.square(y1- y2))
print('Avg RMSE:', np.average(rmse))
rmse = round(np.average(rmse), 2)

text1 = "R2:"+str(r2)+"  RMSE:"+str(rmse)

plt.figure(figsize=(5, 3), dpi=80)
plt.scatter(y1, y2)
plt.title('Neural Network Predictions')
p1, p2 = [0, max(y1)], [0, max(y1)]
plt.plot(p1, p2, color ='red')
plt.text(0, 750, text1, fontsize=12)
plt.xlabel('y_test')
plt.ylabel('pred')
#plt.savefig(plot_path+'11.png')
plt.show()


#=======================LINEAR REGRESSION=====================================================


linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_pred = linreg.predict(x_test)
pred1 = [item for sublist in y_pred for item in sublist]
pred2= np.array(pred1)



plt.figure(figsize=(5, 3), dpi=80)
plt.scatter(y_test, pred2)
plt.title('Linear Regression Predictions')
p1, p2 = [0, max(y_test)], [0, max(y_test)]
plt.plot(p1, p2, color ='red') 
plt.xlabel('log(y_test)')
plt.ylabel('log(pred)')
plt.savefig(plot_path+'51.png')
plt.show()


# calculate Pearson's correlation
corr, _ = pearsonr(y_test, pred2)
print('Pearsons correlation: %.3f' % corr)
rmse= np.sqrt(np.square(y_test- pred2))
print('Avg RMSE:', np.average(rmse))


y1 = np.exp(y_test)-1
y2 = np.exp(pred2)-1


r2 = r2_score(y1, y2)
r2 =round(r2,2)


print('Pearsons correlation: %.2f' % corr)
rmse= np.sqrt(np.square(y1- y2))
print('Avg RMSE:', np.average(rmse))
rmse = round(np.average(rmse), 2)

text1 = "R2:"+str(r2)+"  RMSE:"+str(rmse)

plt.figure(figsize=(5, 3), dpi=80)
plt.scatter(y1, y2)
plt.title('Linear Regression Predictions')
p1, p2 = [0, max(y1)], [0, max(y1)]
plt.plot(p1, p2, color ='red')
plt.text(400, 1200, text1, fontsize=12)
plt.xlabel('y_test')
plt.ylabel('pred')
plt.savefig(plot_path+'51.png')
plt.show()


#====================RIDGE REGRESSION==============================================

# define model
ridge = Ridge(alpha=1.0)
# fit model
ridge.fit(x_train, y_train)

yhat = ridge.predict(x_test)

pred1 = [item for sublist in yhat for item in sublist]
pred2= np.array(pred1)

y1 = np.exp(y_test)-1
y2 = np.exp(pred2)-1


r2 = r2_score(y1, y2)
r2 =round(r2,2)

corr, _ = pearsonr(y1, y2)
corr = round(corr, 2)
print('Pearsons correlation: %.2f' % corr)
rmse= np.sqrt(np.square(y1- y2))
print('Avg RMSE:', np.average(rmse))
rmse = round(np.average(rmse), 2)

text1 = "Corr:"+str(corr)+"  RMSE:"+str(rmse)

plt.figure(figsize=(5, 3), dpi=80)
plt.scatter(y1, y2)
plt.title('Ridge Regression Predictions')
p1, p2 = [0, max(y1)], [0, max(y1)]
plt.plot(p1, p2, color ='red')
plt.text(400, 1200, text1, fontsize=12)
plt.xlabel('y_test')
plt.ylabel('pred')
plt.savefig(plot_path+'52.png')
plt.show()




