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
from sklearn import cross_validation
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# encoding
from sklearn import preprocessing


df = pd.read_csv('real_dataset_321.csv')
df.drop(['Server_time'],1,inplace=True)
df.drop(['ax'],1,inplace=True)
df.drop(['ay'],1,inplace=True)
df.drop(['az'],1,inplace=True)

df.loc[df['localDay'] == 1,'localDay'] = 'Weekday'
df.loc[df['localDay'] == 2,'localDay'] = 'Weekday'
df.loc[df['localDay'] == 3,'localDay'] = 'Weekday'
df.loc[df['localDay'] == 4,'localDay'] = 'Weekday'
df.loc[df['localDay'] == 5,'localDay'] = 'Weekday'
df.loc[df['localDay'] == 6,'localDay'] = 'Weekend'
df.loc[df['localDay'] == 7,'localDay'] = 'Weekend'

df.loc[df['safe'] == True,'safe'] = 1
df.loc[df['safe'] == False,'safe'] = 0

#removal of e.g. row 3783, 3860, empty lines
df['wifi mac'].fillna('NotConnected',inplace=True)


df.to_csv("real_dataset_321_preprocessed.csv", index=False)
