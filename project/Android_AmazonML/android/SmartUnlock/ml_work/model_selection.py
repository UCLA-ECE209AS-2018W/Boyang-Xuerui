# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import datetime
import time
import pytz

# import necessary model to use
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_predict
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
# the error metric, use c-stat, one way to score our model's performance
from sklearn.metrics import roc_auc_score
# encoding
from sklearn import preprocessing

# import data, set x,y
df = pd.read_csv('real_dataset_319_preprocessed.csv')
df = df.reset_index(drop=True)
header = list(df.columns.values)

# e.g. for testing
df.iloc[-1] = ['Weekday','1','10.032324','34.0321','-118.43232','21.4324','0','0','2c:b7:8c:i9:70:1c','"neslwifi"','-64','network','10']

# encode
df = df.apply(preprocessing.LabelEncoder().fit_transform)
new_predict = df.iloc[-1][:-1]
df = df[:-1]

# pop y
y = df.pop(header[-1])
new_y = [i for i in y]




def randomf(df,y):
	# # build model
	# model1 = RandomForestRegressor(n_estimators=20, oob_score=True, random_state=42)
	# model1.fit(df,y) #train all x and y, num and category

	# score = roc_auc_score(y,model1.oob_prediction_)

	# return score

	# hold out validation
	X_train, X_test, y_train, y_test = train_test_split(df,y,test_size=0.3,random_state=42)
	model = RandomForestRegressor(n_estimators=300, oob_score=False, random_state=42)
	model.fit(X_train, y_train)
	
	# single_predict = model.predict([new_predict])

	# if single_predict < 0.8:
	# 	return '0' #not safe

 	# mat = confusion_matrix(y_test, ypred)
	score = model.score(X_test,y_test)
	return score

def knn_c(df,y):

	# # hold out
	# X_train, X_test, y_train, y_test = train_test_split(df,y,test_size=0.3,random_state=42)
	# model = KNeighborsClassifier(n_neighbors=10)
	# model.fit(X_train, y_train)
	# y_pred = model.predict(X_test)
	# accuracy = metrics.accuracy_score(y_test, y_pred)
	# single_predict = model.predict([new_predict])
	# # return accuracy
	# return single_predict, accuracy

	# cross validation
	model = KNeighborsClassifier(n_neighbors=20)
	predicted = cross_val_predict(model, df, y, cv=10)
	score = metrics.accuracy_score(y, predicted)
	return score

def nb_c(df,y):

 #    # hold out
	# X_train, X_test, y_train, y_test = train_test_split(df,y,test_size=0.3,random_state=42)
	# gnb = GaussianNB()
	# gnb.fit(X_train, y_train)
	# y_pred = gnb.predict(X_test)
	# accuracy = metrics.accuracy_score(y_test, y_pred)

	# single_predict = gnb.predict([new_predict])
	# return single_predict

	# return accuracy

	# cross validation
	gnb = GaussianNB()
	predicted = cross_val_predict(gnb, df, y, cv=10)
	score = metrics.accuracy_score(y, predicted)
	return score

# better logistic regression with 10-fold cross validation
def lr_c(df,y):
	model = LogisticRegression()
	predicted = cross_val_predict(model, df, y, cv=10)
	score = metrics.accuracy_score(y, predicted)
	return score


print randomf(df,new_y)
print knn_c(df,new_y)
print nb_c(df,new_y)
print lr_c(df,new_y)


