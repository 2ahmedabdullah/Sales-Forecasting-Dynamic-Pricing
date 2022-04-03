#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 21:46:20 2022

@author: abdul
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


sales = pd.read_csv('sales_small.csv')
sales = sales.rename(columns={'sales_small.ProductID': 'ProductID'})

list(sales)

# Groupby Product ID Total Sales and Stock Volumes

all_prod = sales.groupby(['ProductID']).sum()[['sales_small.SalesVolume', 'sales_small.TotalStockVolume']]
all_prod_avg_price = sales.groupby(['ProductID']).mean()[['sales_small.CSP']]


#Top 10 Selling Products

top_10 = all_prod.sort_values(by=['sales_small.SalesVolume'], ascending = False).head(10)
top_10['ProductID'] = top_10.index

top_10.plot(kind='bar' , x='ProductID', title='10 Most Bought Products', figsize= (6,4), rot = 30)

top_10_products = list(top_10.index)


# Bottom 10 Products

bottom_10 = all_prod.sort_values(by=['sales_small.SalesVolume'], ascending = True).head(10)
bottom_10['ProductID'] = bottom_10.index

bottom_10.plot(kind='bar' , x='ProductID', title='10 Least Bought Products', figsize= (6,4), rot = 30)

bottom_10_products = list(bottom_10.index)



# Price comparison between Top and Bottom products

top_mean_price = all_prod_avg_price.loc[top_10_products, :].mean()[0]
bottom_mean_price = all_prod_avg_price.loc[bottom_10_products, :].mean()[0]

c1 = {'10 Most':top_mean_price, '10 Least':bottom_mean_price}
c1 = pd.Series(c1)

c1.plot(kind='bar' , x='ProductID', title='Avg Price Comparison of Products', figsize= (6,4), rot = 0)



# Computing Weeks
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


# Availablity Comparison Stock Volume with time

fig, ax = plt.subplots(figsize=(6, 4))
ax.legend(top_10_products)

for j in range(0, len(top_10_products)):

    prod2 = sales[sales['ProductID']==top_10_products[j]]
    prod1 = prod2.groupby(['week']).sum()[['sales_small.SalesVolume', 'sales_small.TotalStockVolume']]
    prod1 = prod1.rename(columns={'sales_small.TotalStockVolume': top_10_products[j]})
    prod1['week'] = list(prod1.index) 
    
    prod1.plot(x='week', y=[top_10_products[j]], 
               kind= 'line', title = '10 Most Buying Products Total Stock Volume wrt Time', ax=ax)    


fig, ax = plt.subplots(figsize=(6, 4))
ax.legend(bottom_10_products)

for j in range(0, len(bottom_10_products)):

    prod2 = sales[sales['ProductID']==bottom_10_products[j]]
    prod1 = prod2.groupby(['week']).sum()[['sales_small.SalesVolume', 'sales_small.TotalStockVolume']]
    prod1 = prod1.rename(columns={'sales_small.TotalStockVolume': bottom_10_products[j]})
    prod1['week'] = list(prod1.index) 
    
    prod1.plot(x='week', y=[bottom_10_products[j]], 
               kind= 'line', title = '10 Least Buying Products Total Stock Volume wrt Time', ax=ax)    

'''
late entrance in the market for bottom10 products
'''
'''
these products didn't sell bcoz:
1) new in the market (57th week))
2) may be not popular (brand value, competition, similar products at cheaper price)) 
3) Not enough marketing on the Grocery App/Website
4) these products are not for daily/frequently use like spice/salt
5) Less stock Volume
6) AMazon vs snapdeal (authenticity)
    
'''
'''
the price reduces with time bcoz
these are grocery products and the freshness might be a concern
reaching expiry date
'''
# Top 10 products Sales with time

fig, ax = plt.subplots(figsize=(6, 4))
ax.legend(top_10_products)

for j in range(0, len(top_10_products)):

    prod2 = sales[sales['ProductID']==top_10_products[j]]
    prod1 = prod2.groupby(['week']).sum()[['sales_small.SalesVolume', 'sales_small.TotalStockVolume']]
    prod1 = prod1.rename(columns={'sales_small.SalesVolume': top_10_products[j]})
    prod1['week'] = list(prod1.index) 
    
    prod1.plot(x='week', y=[top_10_products[j]], 
               kind= 'line', title = '10 Most bought Products Sales wrt Time', ax=ax)    


#historical prices of top_10 products with time

fig, ax = plt.subplots(figsize=(6, 4))
ax.legend(top_10_products)

for j in range(0, len(top_10_products)):
    
    prod2 = sales[sales['ProductID']==top_10_products[j]]
    prod1 = prod2.groupby(['week']).mean()[['sales_small.CSP']]
    prod1 = prod1.rename(columns={'sales_small.CSP': top_10_products[j]})
    prod1['week'] = list(prod1.index) 
    
    prod1.plot(x='week', y=[top_10_products[j]], 
               kind= 'line', title = '10 Most bought Products Price wrt Time', ax=ax)    



# price vs Sales of top_10 products LINE

fig, ax = plt.subplots(figsize=(6, 4))
ax.legend(top_10_products)

for j in range(0, len(top_10_products)):
    
    prod2 = sales[sales['ProductID']==top_10_products[j]]
    prod2['CSP'] = prod2['sales_small.CSP'].round(0)

    prod1 = prod2.groupby(['CSP']).mean()[['sales_small.SalesVolume']]
    prod1 = prod1.rename(columns={'sales_small.SalesVolume': top_10_products[j]})
    prod1['CSP'] = list(prod1.index) 
    
    prod1.plot(x='CSP', y=top_10_products[j], 
               kind= 'line', title = '10 Most bought Products Sales vs Price', ax=ax)    


# price vs Sales of top_10 products BAR CHART
for i in range(5, 10):

    prod1 = sales[sales['ProductID']== top_10_products[i]][['sales_small.CSP', 'sales_small.SalesVolume', 'week']]
    prod2 = prod1.groupby(['sales_small.CSP']).sum()[['sales_small.SalesVolume']]
    prod2['CSP'] = list(prod2.index)
    
    plt.bar(x=prod2['CSP'], height=prod2['sales_small.SalesVolume'])   
    plt.xlabel('Price') 
    plt.ylabel('Sales Volume') 
    plt.title(top_10_products[i]) 
    plt.show()


#==================================================================================


# Key findings
'''
TotalStockVolume  = StoreStockVolume + DepotStockVolume	+ FutureCommitmentVolume + IntakeVolume
'''

# online vs Store sales
channels = sales.groupby(['sales_small.Channel']).sum()[['sales_small.SalesVolume']]
channels = pd.Series(channels['sales_small.SalesVolume'])

channels.plot(kind='bar' , x='channels', title='Sales Comparison', figsize= (5,4), rot = 0)


# online vs Store sales
country = sales.groupby(['sales_small.Country']).sum()[['sales_small.SalesVolume']]
country = pd.Series(country['sales_small.SalesVolume'])

country.plot(kind='bar' , x='channels', title='Sales Comparison', figsize= (5,4), rot = 0)


# Top 10 products purchased online

prod2 = sales[sales['sales_small.Channel']=='Online']
prod1 = prod2.groupby(['ProductID']).sum()[['sales_small.SalesVolume', 'sales_small.TotalStockVolume']]


top_10_online = prod1.sort_values('sales_small.SalesVolume', ascending = False).head(10)
top_10_online.plot.bar(rot=30, title='Top 10 Online Selling Products', figsize=(10, 4))




# PRODUCTS

#=====================merged=====================================

products = pd.read_csv('products.csv')
products = products.rename(columns={'products.ProductID': 'ProductID'})

merged = pd.merge(sales, products, on='ProductID')


# All products histroy

prod1 = merged.groupby(['week']).sum()[['sales_small.TotalStockVolume','sales_small.SalesVolume']]
prod1['week'] = list(prod1.index) 

prod1.plot(x='week', y=['sales_small.TotalStockVolume','sales_small.SalesVolume'], 
           kind= 'line', title = 'All Products: Stock Volume & Sales Volume over Time', figsize=(6, 4))    



prod1 = merged.groupby(['week']).mean()[['sales_small.CSP']]
prod1['week'] = list(prod1.index) 

prod1.plot(x='week', y=['sales_small.CSP'], kind= 'line', title = 'All product Price wrt Time', figsize=(6, 4))    


# prduct price with time


prod1 = merged[merged['ProductID']== 'e0fb3756'][['sales_small.CSP', 'sales_small.TotalStockVolume']]
prod1['sales_small.CSP'] = prod1['sales_small.CSP'].round(1)


prod2 = prod1.groupby(['sales_small.CSP']).sum()[['sales_small.TotalStockVolume']]
prod2['sales_small.CSP'] = list(prod2.index) 

prod2.plot.scatter(x=['sales_small.CSP'], y='sales_small.TotalStockVolume') 



# group wise Sales vs price

my_groups = list(merged['products.Group'].unique())

fig, ax = plt.subplots(figsize=(6, 4))
ax.legend(my_groups)

for j in range(0, len(my_groups)):

    prod2 = merged[merged['products.Group']==my_groups[j]]
    prod2['CSP'] = prod2['sales_small.CSP'].round(0)

    prod1 = prod2.groupby(['CSP']).sum()[['sales_small.SalesVolume']]
    prod1 = prod1.rename(columns={'sales_small.SalesVolume': my_groups[j]})
    prod1['CSP'] = list(prod1.index) 
    
    prod1.plot(x='CSP', y = my_groups[j], 
               kind= 'line', title = 'Group wise Products Sales vs Price', ax=ax)   


# All product price recommendation
    
my_products = list(merged['ProductID'].unique())


e =[]
for i in range(0, len(my_products)):
    
    prod1 = merged[merged['ProductID'] == my_products[i]][['sales_small.CSP', 'sales_small.SalesVolume', 'week', 'products.Group']]
    grp = prod1['products.Group'].iloc[0]
    prod1['CSP'] = prod1['sales_small.CSP'].round(1)

    prod2 = prod1.groupby(['CSP']).sum()[['sales_small.SalesVolume']]
    prod2['CSP'] = list(prod2.index)
    prod2['Revenue'] = prod2['CSP'] * prod2['sales_small.SalesVolume']
    
    
    pct_price = (prod2['CSP']-prod2['CSP'].iloc[0])/prod2['CSP'].iloc[0]*100
    pct_qty = (prod2['sales_small.SalesVolume']-prod2['sales_small.SalesVolume'].iloc[0])/prod2['sales_small.SalesVolume'].iloc[0]*100
    
    elasticity = pd.DataFrame(columns = ['%CSP', 'Revenue'])

    elasticity['%CSP'] = pct_price
    elasticity['Revenue'] = prod2['Revenue']
    elasticity['grp'] = [grp]*len(pct_qty)
    
    elasticity.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    elasticity.dropna(inplace = True)
    e.append(elasticity)
    print(i)


e_all = pd.concat(e)
e_all['%CSP'] = e_all['%CSP'].round(0)

e_all1 = e_all.groupby(['%CSP']).mean()[['Revenue']]
e_all1['%CSP'] = list(e_all1.index)

e_all1.plot(x='%CSP', y = '%Qty', 
                title = 'Mean % Change in Qty with % change in CSP: All Products', xlim = [-5,150])   




e_all1 = e_all[e_all['%Qty']<300]

plt.figure(figsize=(6, 4), dpi=80)
num_colors = 1
cmap = plt.get_cmap('Blues', num_colors)
cmap.set_under('red')
plt.scatter(e_all1['%CSP'], e_all1['%Qty'], c =e_all1['%Qty'], cmap=cmap, vmin=0, vmax=1)
plt.title('All products: % Change of Price vs % Qty')
plt.xlabel('% Change CSP')
plt.ylabel('% Change Qty')
plt.show()



# PER product price recommendation


r = random.randint(0, len(my_products))
print(r)
my_products[r]

prod1 = merged[merged['ProductID'] == '3b325781'][['sales_small.CSP', 'sales_small.SalesVolume', 'week']]
prod2 = prod1.groupby(['sales_small.CSP']).sum()[['sales_small.SalesVolume']]
prod2['CSP'] = list(prod2.index)
prod2['Revenue'] = prod2['CSP'] * prod2['sales_small.SalesVolume']

pct_price = (prod2['CSP']-prod2['CSP'].iloc[0])/prod2['CSP'].iloc[0]*100
pct_qty = (prod2['sales_small.SalesVolume']-prod2['sales_small.SalesVolume'].iloc[0])/prod2['sales_small.SalesVolume'].iloc[0]*100

elasticity = pd.DataFrame(columns = ['%CSP', 'Revenue'])

elasticity['%CSP'] = pct_price
elasticity['Revenue'] = prod2['Revenue']

elasticity.replace([np.inf, -np.inf], np.nan, inplace=True)
elasticity.dropna(inplace = True)

fig, ax = plt.subplots(figsize=(5, 4))
elasticity.plot(kind = 'line', x = '%CSP', y = 'Revenue', ax = ax)
plt.title('% Change Price vs Revenue of Product: '+'3b325781')
plt.show()



optimum = []
for j in range(0, len(my_products)):
    prod1 = merged[merged['ProductID'] == my_products[j]][['sales_small.CSP', 'sales_small.SalesVolume', 'week']]

    prod2 = prod1.groupby(['sales_small.CSP']).sum()[['sales_small.SalesVolume']]
    prod2['CSP'] = list(prod2.index)
    
    pct_price = (prod2['CSP']-prod2['CSP'].iloc[0])/prod2['CSP'].iloc[0]*100
    pct_qty = (prod2['sales_small.SalesVolume']-prod2['sales_small.SalesVolume'].iloc[0])/prod2['sales_small.SalesVolume'].iloc[0]*100
    
    elasticity = pd.DataFrame(columns = ['%CSP', '%Qty'])
    
    elasticity['%CSP'] = pct_price
    elasticity['%Qty'] = pct_qty
    
    elasticity.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    elasticity.dropna(inplace = True)
    
    if len(elasticity)>1:
        optimal_increment = elasticity[elasticity['%Qty']== max(elasticity['%Qty'])]
        optimal_increment['ProductID'] = my_products[j] 
        optimum.append(optimal_increment)
    print(j)
    
opti = pd.concat(optimum)

opti1 = opti[opti['%Qty']<10000]


plt.figure(figsize=(10, 4), dpi=80)
plt.scatter(opti1['%CSP'], opti1['%Qty'])
plt.title('Optimum Selling points of Products for Maximum Gains')
plt.xlabel(' % CSP')
plt.show()



# Sales at Group and Class Level
    
prod2 = merged.groupby(['products.Group']).sum()[['sales_small.SalesVolume', 'sales_small.TotalStockVolume']]
prod1 = merged.groupby(['products.Group']).mean()[['sales_small.CSP']]
prod2.plot.bar(rot=30, title='Groups wise Sales', figsize=(6, 4))
prod1.plot.bar(rot=30, title='Groups wise CSP', figsize=(6, 4), width = 0.25)

'''
imported products (kiwi)
less consumable products
'''

prod2 = merged.groupby(['products.Class']).sum()[['sales_small.SalesVolume', 'sales_small.TotalStockVolume']]
prod1 = merged.groupby(['products.Class']).mean()[['sales_small.CSP']]
prod2.plot.bar(rot=30, title='Class wise Sales', figsize=(15, 4))
prod1.plot.bar(rot=30, title='Class wise Price', figsize=(15, 4), width = 0.25)



# history Groupwise Sales wrt time

fig, ax = plt.subplots(figsize=(10, 4))
ax.legend(my_groups)

for j in range(0, len(my_groups)):

#    prod3 = merged[merged['sales_small.Country']=='A']
    prod2 = merged[merged['products.Group']==my_groups[j]]
    prod1 = prod2.groupby(['week']).sum()[['sales_small.SalesVolume', 'sales_small.TotalStockVolume']]
    prod1 = prod1.rename(columns={'sales_small.SalesVolume': my_groups[j]})
    prod1['week'] = list(prod1.index) 
    
    prod1.plot(x='week', y=[my_groups[j]], 
               kind= 'line', title = 'Groupwise Sales wrt Time', ax=ax)    


'''
check local vs global event
Country A vs B
'''


# history Groupwise price wrt time

fig, ax = plt.subplots(figsize=(10, 4))
ax.legend(my_groups)

for j in range(0, len(my_groups)):

    prod2 = merged[merged['products.Group']==my_groups[j]]
    prod1 = prod2.groupby(['week']).mean()[['sales_small.CSP']]
    prod1 = prod1.rename(columns={'sales_small.CSP': my_groups[j]})
    prod1['week'] = list(prod1.index) 
    
    prod1.plot(x='week', y=[my_groups[j]], 
               kind= 'line', title = 'Groupwise Price wrt Time', ax=ax)    




# PRICE STRATEGY
# All product plot
prod1 = merged.groupby(['ProductID']).mean()[['sales_small.CSP']]

prod2 = merged.groupby(['ProductID']).sum()[['sales_small.SalesVolume']]
prod2['CSP'] = prod1['sales_small.CSP']
prod2 = prod2.sort_values('CSP', ascending =False)
prod2 = prod2.rename(columns = {'sales_small.SalesVolume': 'Sales'})
prod2['Profit'] = prod2['CSP']*prod2['Sales']
prod2['CSP'] = prod2['CSP'].round(1)

plt.figure(figsize=(6, 4), dpi=80)
plt.scatter(x=prod2['CSP'], y=prod2['Sales'])   
plt.ylabel('Sales Volume') 
plt.xlabel('CSP') 
plt.title('All products') 
plt.show()


plt.figure(figsize=(6, 4), dpi=80)
plt.scatter(x=prod2['CSP'], y=prod2['Profit'])   
plt.ylabel('Profit') 
plt.xlabel('Price') 
plt.title('All products') 
plt.show()




