# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:02:53 2019

@author: kishan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


Sale_price = df_train.iloc[:,-1]
print(Sale_price[:5])
df_sl_train = df_train.drop("SalePrice",axis=1)
df_sl_train.shape


new_df = pd.concat([df_sl_train,df_test])
new_df.info()

missing_val_count_by_column = (new_df.isnull().sum())




#num_cols12 = new_df._get_numeric_data().columns

numerical_columns_list = [
'LotFrontage', 
'MasVnrArea', 
'BsmtFinSF1',
'BsmtFinSF2', 
'BsmtUnfSF', 
'TotalBsmtSF', 
'BsmtFullBath', 
'BsmtHalfBath', 
'GarageYrBlt', 
'GarageCars', 
'GarageArea'
]


categorical_coloumns = [
 'BsmtQual',
 'BsmtFinType1',
 'KitchenQual',
 'BsmtCond',
 'GarageFinish',
 'GarageQual',
 'Exterior2nd',
 'MSZoning',
 'Electrical',
 'FireplaceQu',
 'Exterior1st',
 'SaleType',
 'BsmtExposure',
 'SaleCondition',
 'MasVnrType',
 'BsmtFinType2',
 'GarageCond',
 'Functional',
 'LandContour',
 'Utilities',
 'GarageType',

]

for cat_l in categorical_coloumns:
    new_df[cat_l] = new_df[cat_l].fillna('None')

missing_val_count_by_column = (new_df.isnull().sum())

for num_col in numerical_columns_list:
    new_df[num_col] = new_df[num_col].fillna(int(0))

missing_val_count_by_column = (new_df.isnull().sum())

corr = df_train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice


drop_list = ['Alley','PoolQC','MiscFeature','Fence',]

for i in drop_list:
    new_df.drop(i,axis=1,inplace=True)


new_df.drop('Utilities',axis=1,inplace=True)



missing_val_count_by_column = (new_df.isnull().sum())

cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC','KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional',  'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating']

from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(new_df[c].values)) 
    new_df[c] = lbl.transform(list(new_df[c].values))



X =  new_df[:1460].values
y = Sale_price.values
X_test_pred = new_df[1460:]

##########################################################################

# Partitioning of Data 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=365)

# MODEL CREATION
# LOGISTIC REGRESSION

from sklearn import metrics 
from sklearn.linear_model import LogisticRegression 
logreg = LogisticRegression()
X_train.shape
y_train.shape
logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test) 
outcome = y_test
print("Accuracy --> ", logreg.score(X_test, y_test)*100)


prediction = logreg.predict(X_test_pred) 
outcome = prediction
df_test['SalePrice'] = outcome

df_test[['Id', 'SalePrice']].to_csv('C:/Users/kishan/Desktop/kaggle/house-prices-advanced-regression-techniques/sample_submission1.csv', index=False)

#####################################

# XGBRegressor

import matplotlib.pyplot as plt
import xgboost
regressor = xgboost.XGBRegressor(learning_rate = 0.06, max_depth= 8, n_estimators = 350, random_state= 0)
regressor.fit(X_train,y_train)

y_hat = regressor.predict(X_train)

plt.scatter(y_train, y_hat, alpha = 0.2)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.show()

regressor.score(X_train,y_train)

y_hat_test = regressor.predict(X_test)

plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.show()

y_predict = regressor.predict(X_test_pred)
y_predict = np.expm1(y_predict)
y_pred = regressor.predict(X_test_pred.values)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 15)


print(accuracies.mean())
print(accuracies.std())

# PREDICTION SUBMISSION

df_test['SalePrice'] = y_pred
df_test[['Id', 'SalePrice']].to_csv('C:/Users/kishan/Desktop/kaggle/house-prices-advanced-regression-techniques/sample_submission.csv', index=False)



