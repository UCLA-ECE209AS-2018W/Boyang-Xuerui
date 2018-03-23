# -*- coding: utf-8 -*-
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import logging
import time
import argparse
import json

# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import pytz

# import necessary model to use
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_predict
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# encoding
from sklearn import preprocessing


AllowedActions = ['both', 'publish', 'subscribe']

# Custom MQTT message callback
def customCallback(client, userdata, message):
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")


# Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--endpoint", action="store", required=True, dest="host", help="Your AWS IoT custom endpoint")
parser.add_argument("-r", "--rootCA", action="store", required=True, dest="rootCAPath", help="Root CA file path")
parser.add_argument("-c", "--cert", action="store", dest="certificatePath", help="Certificate file path")
parser.add_argument("-k", "--key", action="store", dest="privateKeyPath", help="Private key file path")
parser.add_argument("-w", "--websocket", action="store_true", dest="useWebsocket", default=False,
                    help="Use MQTT over WebSocket")
parser.add_argument("-id", "--clientId", action="store", dest="clientId", default="basicPubSub",
                    help="Targeted client id")
parser.add_argument("-t", "--topic", action="store", dest="topic", default="rpiToPhone", help="Targeted topic")
parser.add_argument("-m", "--mode", action="store", dest="mode", default="both",
                    help="Operation modes: %s"%str(AllowedActions))
parser.add_argument("-M", "--message", action="store", dest="message", default="Hello World!",
                    help="Message to publish")

args = parser.parse_args()
host = args.host
rootCAPath = args.rootCAPath
certificatePath = args.certificatePath
privateKeyPath = args.privateKeyPath
useWebsocket = args.useWebsocket
clientId = args.clientId
topic = args.topic

if args.mode not in AllowedActions:
    parser.error("Unknown --mode option %s. Must be one of %s" % (args.mode, str(AllowedActions)))
    exit(2)

if args.useWebsocket and args.certificatePath and args.privateKeyPath:
    parser.error("X.509 cert authentication and WebSocket are mutual exclusive. Please pick one.")
    exit(2)

if not args.useWebsocket and (not args.certificatePath or not args.privateKeyPath):
    parser.error("Missing credentials for authentication.")
    exit(2)

# Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

# Init AWSIoTMQTTClient
myAWSIoTMQTTClient = None
if useWebsocket:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId, useWebsocket=True)
    myAWSIoTMQTTClient.configureEndpoint(host, 443)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath)
else:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
    myAWSIoTMQTTClient.configureEndpoint(host, 8883)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)

# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec


myAWSIoTMQTTClient.connect()

'''
# Connect and subscribe to AWS IoT
if args.mode == 'both' or args.mode == 'subscribe':
    myAWSIoTMQTTClient.subscribe(topic, 1, customCallback)
time.sleep(2)
'''

# Publish to the topic to 'rpiToPhone' pool
if args.mode == 'both' or args.mode == 'publish':
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
	if single_predict[0] < 0.8:
		return 0 #not safe
	else:
            return 1
    
    result = randomf(df,new_y)
    
    message = {}
    message['result'] = result
    messageJson = json.dumps(message)
    myAWSIoTMQTTClient.publish(topic, messageJson, 1)
    if args.mode == 'both' or 'publish':
        print('Published topic %s: %s\n' % (topic, messageJson))
        print '------------------------------------------------'
