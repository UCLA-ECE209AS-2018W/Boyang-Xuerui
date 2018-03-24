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

## Methodology

In this section, we are going to introduce all the methodologies that we used in our project. As mentioned above, our goal of this project is to minimize the times that a user have to log-in on the basis of true positive surrounding safety check. Therefore, our methodology to implement this goal is firstly training supervised machine learning model that having the data labeled by the user as safe (1) or dangerous(0) along with the selected sensors’ data reading from smartphone as different features, and then use this pre-built user Real-time Safety Check Model to real-timely predict the safety of surrounding environment by reading real-time sensor data from the phone. Finally it uses the safety check result to collaborate with smartphone operating system to manage the screen lock mechanism and try to provide the best user experience of unlocking screen along with acceptable level of phone safety. Two main parts of the methodologies that we utilized are Dataset Collection and Dataset Analysis, which directly decides the accuracy of final prediction results of our model in terms of the user’s safety check in the surrounding environment. All the implementations in the Implementation and Evaluation section depends on these core methodologies. 

### Data Collection

Data is the main part of the project, we collected all the meaningful sensor data from the phone and send it to the dataset. In order to collect a large amount of data for the machine learning algorithm, we developed an automated data collection app that can collect data in the backend every 30 seconds. Since we need to use supervised learning, we develop two button for user to choose from, a “safe” button and a “dangerous” button. When the user thinks his current environment is a “safe” place, such as his bedroom, his office or his car, he can choose the safe mode and all the data collected during this mode period is considered “save”. When the user thinks his current environment contains uncertainty (in a crowded area or in library) he can define this area to be a “dangerous” area. After the phone collected the data every 30 seconds, we make use of the Google Script and Google Spreadsheet to store the updated raw data into a Spreadsheet. 

### Data Analysis

With the data analysis process of our method, also as we mentioned above, we utilize elaborate machine (supervised) learning algorithms to analyze and train the datasets as different features in order to generate Real-time Safety Check Model. The system uses the trained model to real-timely receive the new data from surrounding environment by the sensors of smartphone and evaluate the safety of the surroundings for the phone. Finally, the result of safety check will collaborate to manage the screen lock on smartphone in real time.

### Machine Learning Algorithm Selection

As for the selection of machine learning algorithms for our prediction model, after data collecting, it is time to select elegant machine learning algorithms for predicting accurate results. We have selected a number of popular machine learning classifiers, including Logistic Regression with Stochastic Gradient Descent (SGD), K-Nearest Neighbors, Naive Bayes and Random Forest. Using the collected datasets, we are able to primilarily train these models and see general accuracy score of each model, and evaluate which model(s) is/are the most suitable model(s) to be used in our positioning project. Here, we assume that the audiences have already have the basic knowledges and backgrounds of these machine learning algorithms, so this section will not introduce and conclude what these algorithms are and how they mathematically work. Instead, after feature selection (so we have clean dataset), we are going to provide the accuracy stores of each model we select as our evaluation of prediction model. 

### Feature Selection

As is known, in order to train and obtain better model for our prediction goal, it is really necessary to do the feature selection before precisely estimating models. The list shown below represents our original data features taken from the smartphone sensors in the data collection part.

It can be seen that original features that we selected are a lot. We try to catch as many features as possible to be used in our model, but it is also really crucial to remove trivial/unmeaningful features. So firstly we use observation and manual selection in order to do primary selection. We removes the feature of Server time as it can be entirely represented by the feature of Local day and Local time; we also remove the acceleration values of the phone in x,y,z direction, as we really did not see the important features that they potentially have, but they could also pollute the dataset due to their unmeaningful existence, so we think the calculated total acceleration would have enough information to machine learning prediction models. After the observation step, we selected the above selected machine learning models to be trained with different feature combinations, and then score the trained models, finally select the important features as final model. To implement the feature selections for our datasets, the classes in the sklearn.feature_selection module is a good choice, it can be used for feature selection/dimensionality reduction on sample sets, either to improve accuracy scores of the estimator or to boost their performance on datasets that is high dimensional.

After the model training and evaluation in this step, we found that Bluetooth data in our datasets is relatively trivial, as in our team the only usage of the bluetooth is when we use our car built-in bluetooth while we are driving. However, in this project period, we did not use the car frequently, thus causing this feature becoming trivial and we found it has less affection on the accurate score of each model. Nonetheless, it does not mean that this feature is not meaningful for other users. Note that our system is customized for each smartphone user, so the features that are trivial in our evaluation did not mean the same thing in other users’ cases. We would discuss this in the future work that would stick some basic features and allow users to select their customized features into the model. 

Finally, we obtain our meaningful features for our dataset, which could then be used to train the model and evaluate the results. The final selected features are shown below.

Also, as we mentioned above, the label of our dataset is Safety which is binary, SAFE (1) and DANGEROUS (0). The users will play the role of labeling data in the data collection process.  

### Model Training 

After the feature selection (x), along with labeled target data (y), we are able to do the entire processes of model training and testing, and get the accuracy score of each model and process the assessment. To see each performance of the model, we utilize python scikit-learn machine learning library and train the model locally. Based on our observation and verification, we found split the train-test sets to 0.7-0.3 would generally has the best performance for each model, so we change the test_size=0.3 in the model. For each model that we selected and mentioned in the section of Machine Learning Algorithm Selection, the modules of sklearn has them ready and if we need those, we just import those model modules in the Python code.  

Need to note, in this project and the following results of model accuracy score, the method that we use is accuracy_score in metrics module in scikit-learn, which is accuracy classification score and equivalent to Jaccard similarity coefficient score. The Jaccard index/similarity coefficient, defined as the size of the intersection divided by the size of the union of two label sets, which is used to compare set of predicted labels (predicted label) for the test sample to the corresponding set of labels in y_true (ground truth label).

As for the validation technique, model validation technique is used for assessing how the results of data analysis will generalize to an independent dataset. It is mainly used in settings where the goal is prediction, and process the estimation about how accurately a predictive model will perform in practice. In scikit-learn, it has sufficient validation methods for developers to choose. In our project, we select two candidates, the methods of Holdout and K-fold Cross-Validation, to process the data separation for our datasets to do analysis and evaluation.

As for holdout method, it is a type of simple validation that is isolated and involves a single run. The data points are randomly assigned to two sets s0 and s1, as is known, called the training dataset and the test dataset, respectively, which is dealed by model_selection.train_test_split method of sklearn. The size of each of the sets is arbitrary and randomly picked with the parameter of random_state provided by the train_test_split method. We then train on s0 and test on s1. We test the holdout validation method with the methods we selected in previous section. Agreeing control variable method, we pick train-test split test_size=0.3. The performance result is shown in the table below.

In terms of k-fold cross-validation method, the original sample is randomly split into k equal-sized sub-datasets. Of the k sub-datasets, a single sub-dataset is retained as the validation data for testing the model, and the remaining k − 1 sub-datasets are used for training data. The cross-validation process is then repeated k times (the folds), with each of the k sub-datasets used exactly once as the sets of validation data. The k results from the folds can then be averaged to yield a single estimation/prediction. In our project, we implement the evaluation of 10-fold cross-validation which is commonly used with cross_val_predict (cv=10) method from sklearn.model_selection module. The performance result is shown in the table below. 

From the validation results above, we can see although some accuracy results of these models with Holdout validation are better than that of 10-fold Cross-Validation Method, we believe it is due the train-test data random selection mechanism of holdout method. Therefore it is not reliable and stable enough. However, if we look at the 10-fold Cross-Validation, due its partition mechanism of train-test datasets, the results are more convincing, especially under the large dataset such as our case. Therefore, if the model performs well under 10-fold Cross-Validation, it is robust and stable enough to be selected as our prediction model. Based on the results above, we decided to try Random Forest and Logistic Regression as prediction models in our real-world system implementations. 

## Implementation and Evaluation

After coming up with the implementation methodology of Real-time Safety Check Model, in this part, we will introduce two architectural implementations of the Smart Lock Systems on Smartphone that could help user to verify the phone physical safety of the surrounding environment, which are AWS IoT based system and Amazon Machine Learning and Amazon S3 based system.

### Android App for both implementation

We actually develop three app for this project, one for data collection, one for machine learning for Rpi and another one is for Amazon ML. 

For the data collection part, we use multiple thread that running continuously in the background to collect different data. These different threads can be concluded as Wifi Monitor Thread, Location Listening Thread and Bluetooth Monitor Thread. We also have another thread that automatically send data every 30 seconds to Google Spreadsheet, where our data is stored in this spreadsheet. We also designed two button (safe or dangerous) for user to define the label. 

On the prediction part, we wrote two connection type for two app. One is based on the AWS IoT platform and another one is based on the Amazon Machine Learning platform. once the user switch to the predict mode, the app will automatically collect the current status of the user(all the 
collected data) and send it back to the corresponding destination based on the platform we use. 

