# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:41:21 2020

@author: suyog
"""

#Artificial Neural Network ( ANN ) to predict the customer exit in bank

# Part 1 : Data Pre processing and Splitting of test and train data set

#Import Libraries
import pandas as pd

#Play with dataset
bank = pd.read_csv('Bank_Customer_Exit.csv')
X = bank.iloc[:,3:13]
y = bank.iloc[:,13]

#Convert Categorical feature to dummy variable
states = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

#Drop STATE and GENDER column and add dummy varibles to it
X = X.drop(['Geography','Gender'],axis=1)
X = pd.concat([X,states,gender],axis=1)

#Splitting the dataset to test and train data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 : Making ANN network

#Import Keras Libraries and packagess
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initiate ANN
classifier = Sequential()

#Add the input layer and the first hidden layer
classifier.add(Dense(activation='relu',input_dim=11,units=6,kernel_initializer='uniform'))

#Add the second hidden layer
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))

#Add the output layer
classifier.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))

#Compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)


# Part 3 : Makng the predictions and evaluating the model

#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

















