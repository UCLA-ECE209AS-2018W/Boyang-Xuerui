## Abstract

Smart mobile devices have been widely used in modern society. Its convenience enhance the efficiency of the work as well as bring a new life to people. In order to provide a more secured environment, the OS providers introduced screen lock software that is build on top of OS. Despite the improved feature they added on the mobile devices, its high frequency of asking user to login every time he/she wake up the smartphone still bring inconvenience for the user. This article wants to implement a type of machine learning model based system to minimize the times a user have to log in under true positive surrounding safety check. We tried to use supervised machine learning algorithm with two different algorithms to build a model of user based on sensor data of smartphone, including the current time, accelerate, latitude, longitude, altitude, speed, Wi-fi and bluetooth info. By using these features, we got around 97.5% successful rate (accuracy) under a known safe area and a 80% successful rate under a known dangerous area. 

## Background

As mentioned, smartphones have been widely used. According a Pew Research Center survey conducted in November 2016, roughly three quarters of Americans(77%) are now using smartphones and roughly 51% of Americans own a tablet. Since mobile device is getting more and more related to people’s daily life and is getting more powerful than before, it now contains a lot of personal sensitive data such as email, contacts, messages, photos and videos. Sometimes, the smartphone may store user’s login credentials. Thus, mobile device requires more attention on security mechanism than before in order to defend the increasing attack or unauthorized usage. 

In order to provide a more secured environment, the OS providers introduced screen lock software that is build on top of OS. They request a passphrase to log in back to the phone when the screen is off or when the user lock the phone. This mechanism brings a huge plus on security level of the devices. Apart from passphrase, the vendors also implement variety of log in mechanism such as Pin, Pattern, Face Recognition, fingerprint, Iris Scan, FaceID, and voice recognition in order to make log in simpler and faster. However, despite the improved feature they added on the mobile devices, its high frequency of asking user to log in every time he/she wake up the phone still bring inconvenience to the user. Therefore, a smart unlock feature that can reduce the password request times under safe environment can be a benefit. 

Our goal is to minimize the times a user have to log in under true positive surrounding safety check. By using supervised machine learning, we will build a machine learning model based system that can detect whether the current status of the phone is “SAFE” or not. 

## Related Work

In this section, we are going to discuss about the related work from the previous research papers and related technical platforms for smart lock on smart devices (smartphone) which protects user phone security. These related works help us to understand and bypass more as for the difficulty, bottleneck of this research domain and provide the ideas and related technical support on our system implementations.  

### Previous Research

Prior to our study, we acknowledged that there are some related work that has been done. Google has invented the on-body detection feature that enable a detection of movement, when the people is keep in motion and he close his screen, the phone will remain unlocked until the phone detected that it stops moving. On the other hand, some researcher tries to dynamically adjust the unlock time period setting of an auto-lock scheme based on derived knowledge from past user behaviors, current user activities or current user environment. There are some other researchers that analysis the on-body localization of the phone by using the sensors on the phone. And another group of people track all the user activity (location, sound and even their app activity) to conclude a user behavior for authentication. 

### Related Platform

#### TensorFlow Lite

TensorFlow Lite is a lightweight version of TensorFlow. It enables on-device machine learning inference with low latency and a small binary size, and is designed for mobile and embedded devices. TensorFlow Lite supports a set of core operators, both quantized and float, which have been tuned for mobile platforms. 
TensorFlow Lite was released in late 2017 as developer review, and is currently only supporting a few models from TensorFlow. Therefore, we are not gonna use TensorFlow Lite for our project at this time. 

![TensorFlow example with image recogition](https://github.com/UCLA-ECE209AS-2018W/Boyang-Xuerui/blob/master/pics/motocycle.png)

## Methodology

In this section, we are going to introduce all the methodologies that we used in our project. As mentioned above, our goal of this project is to minimize the times that a user have to log-in on the basis of true positive surrounding safety check. Therefore, our methodology to implement this goal is firstly training supervised machine learning model that having the data labeled by the user as safe (1) or dangerous(0) along with the selected sensors’ data reading from smartphone as different features, and then use this pre-built user **Real-time Safety Check Model** to real-timely predict the safety of surrounding environment by reading real-time sensor data from the phone. Finally it uses the safety check result to collaborate with smartphone operating system to manage the screen lock mechanism and try to provide the best user experience of unlocking screen along with acceptable level of phone safety. Two main parts of the methodologies that we utilized are Dataset Collection and Dataset Analysis, which directly decides the accuracy of final prediction results of our model in terms of the user’s safety check in the surrounding environment. All the implementations in the Implementation and Evaluation section depends on these core methodologies. 

### Data Collection

Data is the main part of the project, we collected all the meaningful sensor data from the phone and send it to the dataset. In order to collect a large amount of data for the machine learning algorithm, we developed an automated data collection app that can collect data in the backend every 30 seconds. Since we need to use supervised learning, we develop two button for user to choose from, a “safe” button and a “dangerous” button. When the user thinks his current environment is a “safe” place, such as his bedroom, his office or his car, he can choose the safe mode and all the data collected during this mode period is considered “save”. When the user thinks his current environment contains uncertainty (in a crowded area or in library) he can define this area to be a “dangerous” area. After the phone collected the data every 30 seconds, we make use of the Google Script and Google Spreadsheet to store the updated raw data into a Spreadsheet. 

### Data Analysis

With the data analysis process of our method, also as we mentioned above, we utilize elaborate machine (supervised) learning algorithms to analyze and train the datasets as different features in order to generate **Real-time Safety Check Model**. The system uses the trained model to real-timely receive the new data from surrounding environment by the sensors of smartphone and evaluate the safety of the surroundings for the phone. Finally, the result of safety check will collaborate to manage the screen lock on smartphone in real time.

### Machine Learning Algorithm Selection

As for the selection of machine learning algorithms for our prediction model, after data collecting, it is time to select elegant machine learning algorithms for predicting accurate results. We have selected a number of popular machine learning classifiers, including Logistic Regression with Stochastic Gradient Descent (SGD), K-Nearest Neighbors, Naive Bayes and Random Forest. Using the collected datasets, we are able to primilarily train these models and see general accuracy score of each model, and evaluate which model(s) is/are the most suitable model(s) to be used in our positioning project. Here, we assume that the audiences have already have the basic knowledges and backgrounds of these machine learning algorithms, so this section will not introduce and conclude what these algorithms are and how they mathematically work. Instead, after feature selection (so we have clean dataset), we are going to provide the accuracy stores of each model we select as our evaluation of prediction model. 

### Feature Selection

As is known, in order to train and obtain better model for our prediction goal, it is really necessary to do the feature selection before precisely estimating models. The list shown below represents our original data features taken from the smartphone sensors in the data collection part.

![Original Features](https://github.com/UCLA-ECE209AS-2018W/Boyang-Xuerui/blob/master/pics/original_feature.png)

It can be seen that original features that we selected are a lot. We try to catch as many features as possible to be used in our model, but it is also really crucial to remove trivial/unmeaningful features. So firstly we use observation and manual selection in order to do primary selection. We removes the feature of Server time as it can be entirely represented by the feature of Local day and Local time; we also remove the acceleration values of the phone in x,y,z direction, as we really did not see the important features that they potentially have, but they could also pollute the dataset due to their unmeaningful existence, so we think the calculated total acceleration would have enough information to machine learning prediction models. After the observation step, we selected the above selected machine learning models to be trained with different feature combinations, and then score the trained models, finally select the important features as final model. To implement the feature selections for our datasets, the classes in the **sklearn.feature_selection** module is a good choice, it can be used for feature selection/dimensionality reduction on sample sets, either to improve accuracy scores of the estimator or to boost their performance on datasets that is high dimensional.

After the model training and evaluation in this step, we found that Bluetooth data in our datasets is relatively trivial, as in our team the only usage of the bluetooth is when we use our car built-in bluetooth while we are driving. However, in this project period, we did not use the car frequently, thus causing this feature becoming trivial and we found it has less affection on the accurate score of each model. Nonetheless, it does not mean that this feature is not meaningful for other users. Note that our system is customized for each smartphone user, so the features that are trivial in our evaluation did not mean the same thing in other users’ cases. We would discuss this in the future work that would stick some basic features and allow users to select their customized features into the model. 

Finally, we obtain our meaningful features for our dataset, which could then be used to train the model and evaluate the results. The final selected features are shown below.

Also, as we mentioned above, the label of our dataset is Safety which is binary, SAFE (1) and DANGEROUS (0). The users will play the role of labeling data in the data collection process.  

### Model Training 

After the feature selection (x), along with labeled target data (y), we are able to do the entire processes of model training and testing, and get the accuracy score of each model and process the assessment. To see each performance of the model, we utilize python scikit-learn machine learning library and train the model locally. Based on our observation and verification, we found split the train-test sets to 0.7-0.3 would generally has the best performance for each model, so we change the *test_size*=0.3 in the model. For each model that we selected and mentioned in the section of Machine Learning Algorithm Selection, the modules of sklearn has them ready and if we need those, we just import those model modules in the Python code.  

Need to note, in this project and the following results of model accuracy score, the method that we use is **accuracy_score** in **metrics** module in scikit-learn, which is accuracy classification score and equivalent to Jaccard similarity coefficient score. The Jaccard index/similarity coefficient, defined as the size of the intersection divided by the size of the union of two label sets, which is used to compare set of predicted labels (predicted label) for the test sample to the corresponding set of labels in **y_true** (ground truth label).

As for the validation technique, model validation technique is used for assessing how the results of data analysis will generalize to an independent dataset. It is mainly used in settings where the goal is prediction, and process the estimation about how accurately a predictive model will perform in practice. In scikit-learn, it has sufficient validation methods for developers to choose. In our project, we select two candidates, the methods of Holdout and K-fold Cross-Validation, to process the data separation for our datasets to do analysis and evaluation.

As for holdout method, it is a type of simple validation that is isolated and involves a single run. The data points are randomly assigned to two sets s0 and s1, as is known, called the training dataset and the test dataset, respectively, which is dealed by **model_selection.train_test_split** method of sklearn. The size of each of the sets is arbitrary and randomly picked with the parameter of random_state provided by the train_test_split method. We then train on s0 and test on s1. We test the holdout validation method with the methods we selected in previous section. Agreeing control variable method, we pick train-test split test_size=0.3. The performance result is shown in the table below.

In terms of k-fold cross-validation method, the original sample is randomly split into k equal-sized sub-datasets. Of the k sub-datasets, a single sub-dataset is retained as the validation data for testing the model, and the remaining k − 1 sub-datasets are used for training data. The cross-validation process is then repeated k times (the folds), with each of the k sub-datasets used exactly once as the sets of validation data. The k results from the folds can then be averaged to yield a single estimation/prediction. In our project, we implement the evaluation of 10-fold cross-validation which is commonly used with **cross_val_predict** (cv=10) method from **sklearn.model_selection** module. The performance result is shown in the table below. 

From the validation results above, we can see although some accuracy results of these models with Holdout validation are better than that of 10-fold Cross-Validation Method, we believe it is due the train-test data random selection mechanism of holdout method. Therefore it is not reliable and stable enough. However, if we look at the 10-fold Cross-Validation, due its partition mechanism of train-test datasets, the results are more convincing, especially under the large dataset such as our case. Therefore, if the model performs well under 10-fold Cross-Validation, it is robust and stable enough to be selected as our prediction model. Based on the results above, we decided to try **Random Forest** and **Logistic Regression** as prediction models in our real-world system implementations. 

## Implementation and Evaluation

After coming up with the implementation methodology of Real-time Safety Check Model, in this part, we will introduce two architectural implementations of the Smart Lock Systems on Smartphone that could help user to verify the phone physical safety of the surrounding environment, which are AWS IoT based system and Amazon Machine Learning and Amazon S3 based system.

### Android App for both implementation

We actually develop three app for this project, one for data collection, one for machine learning for Rpi and another one is for Amazon ML. 

For the data collection part, we use multiple thread that running continuously in the background to collect different data. These different threads can be concluded as Wifi Monitor Thread, Location Listening Thread and Bluetooth Monitor Thread. We also have another thread that automatically send data every 30 seconds to Google Spreadsheet, where our data is stored in this spreadsheet. We also designed two button (safe or dangerous) for user to define the label. 

On the prediction part, we wrote two connection type for two app. One is based on the AWS IoT platform and another one is based on the Amazon Machine Learning platform. once the user switch to the predict mode, the app will automatically collect the current status of the user(all the 
collected data) and send it back to the corresponding destination based on the platform we use. 

### Android App + Server (RPi)  + AWS IoT + Random Forest

In this system implementation, we utilize the implemented Android App to communicate with the built local server under the protocol and policy of AWS IoT service. The local server will handle the Machine Learning Safety Check Model, including data training and label prediction. As mentioned, We mainly implemented machine learning prediction model with Python sklearn library and with the result receiving and sending scripts, the safety check result will finally be sent to the mobile end. More specifically:

As server, in our case, we use Raspberry Pi (could be any PC or cluster). The Rpi will have credential to be a role in AWS IoT, and holding the keys and config issued by AWS. The server will be initialized by the script first, and then the rpi server will subscribe the topic in MQTT protocol, named **phoneToRPi**, supported my AWS IoT. When phone needs to check the surrounding safety, it will send the request to **phoneToRPi** with JSON format. Then the server will load the data from JSON to pre-built safety check model and make the safety prediction. After prediction, the server will send the safety check result in JSON as well to the topic of **rpiToPhone** that the phone is subscribing. The Phone app will then feed the check result and collaborate with operating system to manage the screen lock.  

As for the advantages of this system implementation, it is actually very flexible for more different types of machine learning models, and you have large enough freedom to define your models. Also as you prepare all the materials and devices yourself, it is not necessary for you to pay any extra fee for any other services such as computation via cloud service. 

However, this system do also have its limitations. For instance, it has weak scalability, so you need almost need to redeploy everything again and again if you want to have more systems with different applying environment, as the AWS IoT would limit the certificates so that each thing in the AWS IoT need to have unique ID, which would give rise to the restriction of scalability. Besides, the latency is another issue, in our system, the rpi average response time for each phone request is 3 - 5s, which is too long for user to wait the decision made by phone. Nonetheless, this latency issue could be resolved with the computer or the cluster which have better computing power, so this issue is really computation and computer performance dependent. 

### Android App + Amazon Machine Learning + Amazon S3 + Logistic Regression

As for this system implementation, the preprocessed data are stored in the Amazon S3 as data storage warehouse. Then the developer is able to select the features and tune the parameters for building and training the dataset by using Amazon Machine Learning. After the model is built, we would create real-time prediction endpoint for phone to do safety check request. If there is a check request sent by the phone, Amazon ML will receive the phone check HTTP request through Amazon ML Predict API. The request with feature data will be read and plug into the built prediction Safety Check Model, and yield the prediction result and it to the phone with HTTP response.

We utilized logistic regression algorithm in the prediction model of Amazon ML, though the data we collected is kind of unbalanced, the prediction result shows the model still have really good and reliable accuracy of prediction. As the plot shown below, the SAFE (1) / DANGEROUS (0) conditions/labels are well distinguished and predicted.

As for the advantages of this system implementation, as the cloud service from AWS, it is really easy to be deployed and very salable. In addition, it has very low latency. Based on our test with from phone request to Amazon ML response, it is sub-second level latency. Besides, the system shows really good reliability, and AWS also guarantee the security.  

As for the disadvantages of the system, we have to pay the fee to AWS by the number of predictions and hourly reserved capacity as our service this real-time prediction. In addition, there is the limitation of Amazon ML. We can only train and implement several types of machine learning models including neural network, linear regression, and logistic regression. So the freedom of choice of our prediction models is relatively constrained, but it could be expected AWS would add more model options for developers. 

## Discussion and Future Work

In this section, we will make the comparison of other similar implementation with our research and implementations. Rely on the comparison results, and the problems we found during the research and development, wer will also provide the future work that would help to improve the feasibility, accuracy and reliability of our Smart Lock System on Smartphone. 

### Performance compared with phone’s original feature

In order to evaluate how Smart Lock can help user to save log in times, we develop some test and compare the performance of Smart Lock vs Original phone. 

* The true positive rate is defined when the given current situation is safe and the prediction result is also safe

* The true negative rate is defined when the given current situation is dangerous and the prediction result is also dangerous

From the above data we can see for the true positive check the Location Detection shows a pretty good result. When the tester is staying at home, the model gives all positive result based on user’s location, wifi and bluetooth information. For the original phone, since the original location server is based on the home address, therefore there might be a small bias between the actual location and the latitude + longitude. 

For the in car test, both result shows a pretty strong prediction, the only two missed test comes out where the bluetooth is disconnected from the phone when the tester started the car. Therefore, the SmartLock shows a good result on the in car test. 

Our model does not have a good result on movement test where only 2 out of 40 tests result in a true positive result. This may because that the model only use one state to infer the status where the on-body detection uses the context states to infer the status of the user. The model should be improved by supporting centext states. 

As for the true negative test, we got 32/40 rate, which is approximately 80% rate of successfully predict the dangerous rate. We understand that this rate should be as close as 100% to give the strong security feature, however, implementing a SmartLock is actually a tradeoff between security and convenience. And I think we can improve this rate by changing the threshold in the model on use more features to get to our conclusion. 

### Discussion and Future Work

We finished our project in a very small time period. we found out that there could be a lot improvement in the future for this model. 

For our database, we can track different data in an unseen area, and test the unseen data using the trained data. Therefore, we got the generalized result. Due to the daily activity, our dataset is too skewed to safe area (75% safe area vs 25% dangerous area), causing the percentage of rate meaningless. In the future, we can collect more data in dangerous area to make the data balance between safe area and dangerous area. We also evaluate the model in multiple ways including confusion matrix instead of only getting the visualization results as for how many exact data are predicted wrong. 

We can change our model into a real time based machine learning model. Currently, our model is based on a non-real time model which require us to manually train the model every day in order to cover the updated data. Once we got a data from the phone, there is no way for us to update it to the model immediately. Using real time based machine learning mechanizm can provide a faster response from the real situation as well as save the manually train time. 

Currently, our model is based on the current state of the user data. The model does not took user’s previously input as a reference. Therefore, the model can not make a good prediction on user’s moving data. Once the model can combine the user’s current status with the user’s previous status and took everything into consideration, the result would be more intelligence and more accurate.

Considering the fact that some user may not want us to know “everything” they have, we can provide a feature selection that user can choose from. And make the judgement based on the allowed features that user provided. Insufficient data may result in a increasing on the False Negative rate in order to maintain a small false positive rate. 

## Project Timeline

**Week 6**  

- Set up the project objective and propose the project.

**Week 7**

- Research on different approaches of machine learning algorithms. 
- Implementing the frontend and the backend for the local server (Raspberry Pi) way in AWS IoT.

**Week 8**  

- Start collecting data using data collection app.
- Test the AWS IoT -> Raspberry Pi -> Android App performance.

**Week 9**  

- Collecting data using data collection app.
- Implement a new backend method utilizing Amazon Machine Learning and Amazon S3.
- Implement the corresponding front end and connection to Amazon ML.
- Merge data collection app with prediction app.

**Week 10**  

- Debug and verify the final version of two smart lock systems.
- Prepare for Project PPT/Demo and Final Project Report.

## References

[1] http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7153935&tag=1

Intelligent Display Auto-lock Scheme for Mobile Devices by Nai-Wei Lo

[2] https://link.springer.com/content/pdf/10.1007/978-3-642-41674-3_63.pdf

Smartphone Based Data Mining for Fall Detection: Analysis and Design by Abdul Hakim

[3] http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6258136

Design and Implementation of an On-body Placement-aware Smartphone by Kaori Fujinami

[4] https://www.ncbi.nlm.nih.gov/pubmed/22205862

Machine learning methods for classifying human physical activity from on-body accelerometers

[5] https://www.sciencedirect.com/science/article/pii/S1877050917302065

Smartphone Based Data Mining for Fall Detection: Analysis and Design

[6] http://elaineshi.com/docs/isc.pdf

Implicit Authentication through Learning User Behavior

[7] https://serverlesscode.com/post/deploy-scikitlearn-on-lamba/

Using scikit-learn in AWS Lambda

[8] https://docs.aws.amazon.com/machine-learning/latest/dg/requesting-real-time-predictions.html

Requesting Real-time prediction

[9] https://www.androidpolice.com/2015/03/20/trusted-butts-new-on-body-detection-smart-lock-mode-in-android-seems-to-be-hitting-some-devices/

New “On-Body Detection” Smart Lock Mode in Android Seems To Be Hitting Some Devices
