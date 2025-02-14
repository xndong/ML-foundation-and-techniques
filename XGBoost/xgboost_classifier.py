# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:37:10 2019

@author: DongXiaoning
"""
import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)




import xgboost as xgb
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# 载入数据集
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
 
# 把数据集拆分成训练集和测试集
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
 
# 拟合XGBoost模型
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
 
# 对测试集做预测
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
 
# 评估预测结果
accuracy = accuracy_score(y_test, predictions)
