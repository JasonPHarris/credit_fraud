# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:24:19 2022

@author: Jason Harris
CS379 Machine Learning Due 02/13444444/2022
using creditfraud.arff to create machine learning code to recognize future fraud attempts.
Usaing the following websites to create this code:
https://dataaspirant.com/credit-card-fraud-detection-classification-algorithms-python/#:~:text=The%20decision%20tree%20is%20the,up%20with%20the%20important%20features.
I also used https://pulipulichen.github.io/jieba-js/weka/arff2csv/ to convert the arff to csv for viewing in excel while working with the data
This script is to train and test (and compare) the Decision Tree Classifier and Random Forest Classifier algorithms
in the case of detecting credit card fraud.

"""

from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

data = arff.loadarff('creditfraud.arff')                            # importing the arff
df = pd.DataFrame(data[0])                                          # loading the arff into pandas

print(f"Dataset Shape :- \n {df.shape}")                            # getting the shape of the data frame
print()
print('************************************************')
print()
print(df.dtypes)                                                    # verifying the data types to know what needs to be changed
print()
print('*********************************************')
print()
print(f"Unique values of target variable:- \n {df['class'].unique}")# checking for the unique values in class column to know which will be fraud and non-fraud
print()
print('*********************************************')
print()
# starting encoder to clean up the data types to make the data useable by the algorithms
labelencoder = LabelEncoder()               
df.iloc[:, 20]=labelencoder.fit_transform(df.iloc[:, 20].values)    # changing class to a numerical value 0 = bad 1 = good
df.iloc[:, 3]=labelencoder.fit_transform(df.iloc[:, 3].values)      # changing purpose to a numerical value
df.iloc[:, 2]=labelencoder.fit_transform(df.iloc[:, 2].values)      # changing credit_history to a numerical value
df.iloc[:, 6]=labelencoder.fit_transform(df.iloc[:, 6].values)      # changing employment to a numerical value
df.iloc[:, 5]=labelencoder.fit_transform(df.iloc[:, 5].values)      # changing Average_Credit_Balance to a numerical value
df.iloc[:, 0]=labelencoder.fit_transform(df.iloc[:, 0].values)      # changing over_draft to a numerical value
df.iloc[:, 8]=labelencoder.fit_transform(df.iloc[:, 8].values)      # changing personal_status to a numerical value
df.iloc[:, 9]=labelencoder.fit_transform(df.iloc[:, 9].values)      # changing other_parties to a numerical value
df.iloc[:, 11]=labelencoder.fit_transform(df.iloc[:, 11].values)    # changing property_magnitude to a numerical value
df.iloc[:, 13]=labelencoder.fit_transform(df.iloc[:, 13].values)    # changing other_payment_plans to a numerical value
df.iloc[:, 14]=labelencoder.fit_transform(df.iloc[:, 14].values)    # changing housing to a numerical value
df.iloc[:, 16]=labelencoder.fit_transform(df.iloc[:, 16].values)    # changing job to a numerical value
df.iloc[:, 18]=labelencoder.fit_transform(df.iloc[:, 18].values)    # changing own_telephone to a numerical value
df.iloc[:, 19]=labelencoder.fit_transform(df.iloc[:, 19].values)    # changing foreign_worker to a numerical value
print(f"Unique values of target variable:- \n {df['class'].unique}")# verifying data values were changed to match 0 == bad
print()
print('*********************************************')
print()
print(f"Number of samples under each target value :- \n {df['class'].value_counts()}") # checking the number of good and bad transactions
print()
print('*********************************************')
print()
print(f"Dataset info :- \n {df.info()}")                            # verifying no nan exist and checking data types in each column
print()
print('*********************************************')
print()
df['norm_usage'] = StandardScaler().fit_transform(df['credit_usage'].values.reshape(-1,1))  # normalizing usage data
df = df.drop(['credit_usage'], axis=1)                                                      # dropping usage column for appended normalized data
print(f"few values of the Credit Usage column after applying StandardScaler:- \n {df['norm_usage'][0:4]}") # verification
print()
print('*********************************************')
print()
print(f"Dataset info:- \n {df.info()}")                             # verifying changes
print()
print('*********************************************')
print()
df['norm_balance'] = StandardScaler().fit_transform(df['current_balance'].values.reshape(-1,1)) # normalizing balance data
df = df.drop(['current_balance'], axis=1)                                                       # dropping balance column for appended normalized data
print(f"few values of the Currrent Balance column after applying StandardScaler:- \n {df['norm_balance'][0:4]}") # verification
print()
print('*********************************************')
print()
print(f"Dataset info:- \n {df.info()}")                             # verifying changes
print()
print('*********************************************')
print()
# time to train the data ("Time to make the donuts") because it is what we do!
X = df.drop(['class'], axis = 1)                                    # creating the X axis variable without the 'class' column
Y = df[['class']]                                                   # creating the Y axis variable with only the 'class column
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0) # creating variables
print(X_train.shape)                                                # training X
print(X_test.shape)                                                 # testing X
print(y_train.shape)                                                # training Y
print(y_test.shape)                                                 # testing Y
print()
print('*********************************************')
print()
##### Building Fraud Detection Models #####
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
def decision_tree_classification(X_train, y_train, X_test, y_test): # Creating the Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier()                        # initialize object for DecisionTreeClassifier class                     
    dt_classifier.fit(X_train, y_train.values.ravel())              # train model by using fit method
    print("Model training completed")
    acc_score = dt_classifier.score(X_test, y_test)
    print(f'Accuracy of model on test dataset :- {acc_score}')
    y_pred = dt_classifier.predict(X_test)                                          # predict result using test dataset
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")             # confusion matrix
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")   # classification report for f1-score
    
decision_tree_classification(X_train, y_train, X_test, y_test)      # calling the decision tree to see the results
print()
print('*********************************************')
print()
from sklearn.ensemble import RandomForestClassifier
def random_forest_classifier(X_train, y_train, X_test, y_test):
     rf_classifier = RandomForestClassifier(n_estimators=50)        # initialize object for DecisionTreeClassifier class                       
     rf_classifier.fit(X_train, y_train.values.ravel())             # train model by using fit method again
     acc_score = rf_classifier.score(X_test, y_test)
     print(f'Accuracy of model on test dataset :- {acc_score}')
     y_pred = rf_classifier.predict(X_test)                                         # predict result using test dataset
     print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")            # confusion matrix
     print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")  # classification report for f1-score

random_forest_classifier(X_train, y_train, X_test, y_test)          # calling random_forest_classifier
print()
print('*********************************************')
print()
class_val = df['class'].value_counts()
print(f"Number of samples for each class : \n {class_val}")
non_fraud = class_val[1]
fraud = class_val[0]
print(f"Non Fraudulent Numbers : {non_fraud}")
print(f"Fraudulent Numbers : {fraud}")
print()
print('*********************************************')
print()
non_fraud_indexies = df[df['class'] == 1].index                     # indexies of non_fraudulent transactions        
fraud_indexies = np.array(df[df['class'] == 0].index)               # indexies of fraudulent transactions
random_normal_indexies = np.random.choice(non_fraud_indexies, fraud, replace=False) # random samples from nonfraud that are equal to fraud
random_normal_indexies = np.array(random_normal_indexies)

under_sample_indexies = np.concatenate([fraud_indexies, random_normal_indexies])    # concatination of indexies
under_sample_data = df.iloc[under_sample_indexies, :]                               # extract all features from whole data in under_sample_indexies
x_undersample_data = under_sample_data.drop(['class'], axis = 1)                    # assigning under sampled data to the x axis minus the class data
y_undersample_data = under_sample_data[['class']]                                   # assinging under sampled 'class' data to y axis
# splitting the dataset into train and test datasets
X_train_sample, X_test_sample, Y_train_sample, Y_test_sample = train_test_split(x_undersample_data, y_undersample_data, test_size=0.2, random_state=0)

# creating a new decision tree on under sampled data
from sklearn.metrics import roc_auc_score

def decision_tree_classification(X_train, Y_train, X_test, Y_test):
    dt_classifier = DecisionTreeClassifier()                             # defining the classifier class
    dt_classifier.fit(X_train, Y_train.values.ravel())                   # yet again, training using fit
    acc_score = dt_classifier.score(X_test, Y_test)                      # creating the accuracy score variable
    print(f"Accuracy score results: \n {acc_score}")                     # printing the accuracy score results
    print()
    print('*********************************************')
    print()
    y_pred = dt_classifier.predict(X_test)                                       # creating a prediction
    print(f"Decision Tree Confusion Martic: \n {confusion_matrix(Y_test, y_pred)}")           # printing the Confusion Matrix
    print()
    print('*********************************************')
    print()
    print(f"Decision Tree Classification Report: \n {classification_report(Y_test, y_pred)}") # printing the Classification Report
    print()
    print('*********************************************')
    print()
    print(f"Decision Tree AROC Score: \n {roc_auc_score(Y_test, y_pred)}")                    # printing the AROC Score Results
    print()
    print('*********************************************')
    print()
    
decision_tree_classification(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample) # calling the decision tree function

def random_forest_classifier(X_train, y_train, X_test, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=50)             # initialize object for DecisionTreeClassifier class
    rf_classifier.fit(X_train, y_train.values.ravel())                  # train model by using fit method
    acc_score = rf_classifier.score(X_test, y_test)
    print(f'Accuracy of model on test dataset :- {acc_score}')
    print()
    print('*********************************************')
    print()
    y_pred = rf_classifier.predict(X_test)                                          # predict result using test dataset
    print(f"Random Forest Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")             # confusion matrix
    print()
    print('*********************************************')
    print()
    print(f"Random Forest Classification Report :- \n {classification_report(y_test, y_pred)}")   # classification report for f1-score
    print()
    print('*********************************************')
    print()
    print(f"Random Forest AROC score :- \n {roc_auc_score(y_test, y_pred)}")                      # area under roc curve
    print()
    print('*********************************************')
    print()
    
random_forest_classifier(X_train_sample, Y_train_sample, X_test_sample, Y_test_sample)
