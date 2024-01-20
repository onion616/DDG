# -*- coding: utf-8 -*-
"""
__author__ = 'onion'
"""

#导入类库和加载数据集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import ensemble
import xgboost as xgb
import csv


#读取数据
train_names = ["size",
               "focal_age",
               "focal_star",
               "focal_fork",
               "focal_rank",
               "focal_keywords",
               "focal_dependent packages",
               "focal_dependent repositories",
               "focal_versions",
               "downstream_age",
               "downstream_star",
               "downstream_fork",
               "downstream_rank",
               "downstream_keywords",
               "downstream_dependent packages",
               "downstream_dependent repositories",
               "downstream_versions"
               ]
data = pd.read_csv("./data/RubyGems/predict_cddg_feature_less100.csv", header=0, names=train_names)
# print(data.head())
# print(data.info())

# #观察package size的数据分布
# plt.figure(figsize=(10, 5))
# # plt.xlabel('price')
# sns.displot(data['downstream_versions'])
# # plt.show()

#自变量与因变量的相关性分析
plt.figure(figsize=(30, 25))
internal_chars = ["size", "focal_age", "focal_star", "focal_fork", "focal_rank", "focal_keywords",
                  "focal_dependent packages", "focal_dependent repositories", "focal_versions",
                  "downstream_age", "downstream_star", "downstream_fork", "downstream_rank", "downstream_keywords",
                  "downstream_dependent packages", "downstream_dependent repositories", "downstream_versions"]
corrmat = data[internal_chars].corr()  # 计算相关系数
# sns.heatmap(corrmat, square=False, linewidths=.5, annot=True) #热力图
# plt.savefig('./data/CPAN/picture/heatmap.png')
# plt.show()

print(data)
#打印出相关性的排名
# print(corrmat["downstream_versions"].sort_values(ascending=False))

#特征缩放；归一化
data = data.astype('float')
x = data.drop('size', axis=1)
y = data['size']
scaler = MinMaxScaler()
newX = scaler.fit_transform(x)
newX = pd.DataFrame(newX, columns=x.columns)
# print(newX.head())

#将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.2, random_state=21)
# print(y_test)
# test_result = y_test.to_dict()
# print(test_result)


#模型建立
#随机森林
def RF(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=200, max_features=None)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predicted))
    mae = metrics.mean_absolute_error(y_test, predicted)
    mape = metrics.mean_absolute_percentage_error(predicted, y_test)
    return mae, rmse, mape


#线性回归
def LR(X_train, X_test, y_train, y_test):
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    predicted = LR.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predicted))
    mae = metrics.mean_absolute_error(y_test, predicted)
    mape = metrics.mean_absolute_percentage_error(predicted, y_test)
    return mae, rmse, mape

#xgboost
def XGBoost():
    xg_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=10,
        alpha=10
    )
    xg_reg.fit(X_train, y_train)
    pred = xg_reg.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(pred, y_test))
    mae = metrics.mean_absolute_error(pred, y_test)
    mape = metrics.mean_absolute_percentage_error(pred, y_test)
    return mae, rmse, mape


#svm
def SVM():
    model_SVR = svm.SVR()
    model_SVR.fit(X_train, y_train)
    pred = model_SVR.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(pred, y_test))
    mae = metrics.mean_absolute_error(pred, y_test)
    mape = metrics.mean_absolute_percentage_error(pred, y_test)
    return mae, rmse, mape


#knn
def KNN():
    model_KNN = neighbors.KNeighborsRegressor()
    model_KNN.fit(X_train, y_train)
    pred = model_KNN.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(pred, y_test))
    mae = metrics.mean_absolute_error(pred, y_test)
    mape = metrics.mean_absolute_percentage_error(pred, y_test)
    return mae, rmse, mape


#adaboost
def Adaboost():
    model_AdaBoost = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
    model_AdaBoost.fit(X_train, y_train)
    pred = model_AdaBoost.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(pred, y_test))
    mae = metrics.mean_absolute_error(pred, y_test)
    mape = metrics.mean_absolute_percentage_error(pred, y_test)
    return mae, rmse, mape


#GBRT
def GBRT():
    model_GBRT = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
    model_GBRT.fit(X_train, y_train)
    pred = model_GBRT.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(pred, y_test))
    mae = metrics.mean_absolute_error(pred, y_test)
    mape = metrics.mean_absolute_percentage_error(pred, y_test)
    return mae, rmse, mape


print('LR mae, rmse, mape: ', LR(X_train, X_test, y_train, y_test))
print('RF mae, rmse, mape: ', RF(X_train, X_test, y_train, y_test))
print('KNN mae, rmse, mape:', KNN())
print('Adaboost mae, rmse, mape:', Adaboost())
print('GBRT mae, rmse, mape:', GBRT())



