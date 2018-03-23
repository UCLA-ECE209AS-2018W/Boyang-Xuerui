# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import datetime
import time
import pytz
import json

# import necessary model to use
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
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

# read json from phonetorpi poll
with open('input.json') as json_data:
    d = json.load(json_data)

col_order = ["localDay","localTime","g","latitude","longitude","accuracy","altitude","speed","wifi mac","wifi ssid","wifi signal level","provider","safe"]
df.iloc[-1] = [str(d[item]) for item in col_order]

# encode
df = df.apply(preprocessing.LabelEncoder().fit_transform)
new_predict = df.iloc[-1][:-1]
df = df[:-1]

# pop y
y = df.pop(header[-1])
new_y = [i for i in y]


def randomf(df,y):

	# hold out validation
	X_train, X_test, y_train, y_test = train_test_split(df,y,test_size=0.3,random_state=42)
	model = RandomForestRegressor(n_estimators=200, oob_score=False, random_state=42)
	model.fit(X_train, y_train)
	
	single_predict = model.predict([new_predict])
	print single_predict
	if single_predict[0] < 0.5:
		return 0 #not safe
	else:
		return 1

result = randomf(df,new_y)
print result

