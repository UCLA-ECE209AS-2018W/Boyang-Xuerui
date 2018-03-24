# Smart Lock on SmartPhone

Xuerui Yan, Boyang Cai

## Abstract

Smart mobile devices has been widely used in modern society. Its convenience enhance the efficiency of the work as well as bring a new life to people. In order to provide a more secured environment, the OS providers introduced screen lock software that is build on top of OS. Despite the improved feature they added on the mobile devices, its high frequency of asking user to log in every time he/she wake up the phone still bring inconvenience for the user. This article wants to implement a machine learning model to minimize the times a user have to log in under true positive surrounding safety check. We tried to use supervised machine algorithm with two different algorithm to build a model of user based on his/her sersor data, including the current time, accelerate, latitude, longitude, altitude, speed, wifi info and bluetooth info. By using this feature, we got a 97.5% successful rate under a known safe area and a 80% successful rate under a known dangerous area. 

## Background

Smart mobile devices has been widely used in modern society. Its convenience enhance the efficiency of the work as well as bring a new life to people. According a Pew Research Center survey conducted in November 2016, roughly three quarters of Americans(77%) are now using smartphones and roughly 51% of Americans own a tablet. Since mobile device is getting more and more related to people’s daily life and is getting more powerful than before, it now contains a lot of personal sensitive data such as email, contacts, messages, photos and videos. Sometimes, the smartphone may store user’s login credentials. Thus, mobile device requires more attention on security mechanism than before in order to defend the increasing attack or unauthorized usage. 

In order to provide a more secured environment, the OS providers introduced screen lock software that is build on top of OS. They request a passphrase to log in back to the phone when the screen is off or when the user lock the phone. This mechanism brings a huge plus on security level of the devices. Apart from passphrase, the vendors also implement variety of log in mechanism such as Pin, Pattern, Face Recognition, fingerprint, Iris Scan, FaceID, and voice recognition in order to make log in simpler and faster. However, despite the improved feature they added on the mobile devices, its high frequency of asking user to log in every time he/she wake up the phone still bring inconvenience for the user. Therefore, a smart unlock feature that can reduce the password request times under safe environment can be a benefit. 

Our goal is to minimize the times a user have to log in under true positive surrounding safety check. By using supervised machine learning, we will build a model that can detect whether the current status of the phone is “safe” or not. 

## Related Work

In this section, we are going to discuss about the related work from the previous research papers and related technical platforms for smart lock on smart devices (smartphone) which protects user phone security. These related works help us to understand and bypass more as for the difficulty, bottleneck of this research domain and provide the ideas and related technical support on our system implementations.  

### Previous Research

Prior to our study, we acknowledged that there are some related work that has been done. Google has invented the on-body detection feature that enable a detection of movement, when the people is keep in motion and he close his screen, the phone will remain unlocked until the phone detected that it stops moving. On the other hand, some researcher tries to dynamically adjust the unlock time period setting of an auto-lock scheme based on derived knowledge from past user behaviors, current user activities or current user environment. There are some other researchers that analysis the on-body localization of the phone by using the sensors on the phone. And another group of people track all the user activity (location, sound and even their app activity) to conclude a user behavior for authentication. 

### Related Platform

#### TensorFlow Lite

TensorFlow Lite is a lightweight version of TensorFlow. It enables on-device machine learning inference with low latency and a small binary size, and is designed for mobile and embedded devices. TensorFlow Lite supports a set of core operators, both quantized and float, which have been tuned for mobile platforms. 
TensorFlow Lite was released in late 2017 as developer review, and is currently only supporting a few models from TensorFlow. Therefore, we are not gonna use TensorFlow Lite for our project at this time. 
