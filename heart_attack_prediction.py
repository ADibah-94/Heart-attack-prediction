# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:32:04 2022

@author: HP
"""

import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


#%% Paths

MMS_SCALER_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')
OHE_SCALER_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','ohe_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model','model.pkl')
#%% EDA

# Step 1) Data loading

df = pd.read_csv('heart.csv')

# Step 2) Data Inspection

df.head()           # to view the first 5 rows of data
df.info()           # to view the summary of the dataframe
df.describe()       # to view the statistics of data
df.describe().T
df.isnull().sum()   # to check any missing values

# No missing values in data

# Step 3) Data cleaning
# Step 4) Features Selection

df['output'].value_counts() # to check the distribution of Target variables

# 0 --> Negative
# 1 --> Positive

X = df.drop(columns='output', axis=1)   # Features
Y = df['output']                        # Target

print(X)
print(Y)


# Step 5) Data preprocessing

# Perform Min-Max Scaling

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open(MMS_SCALER_SAVE_PATH,'wb'))

# To save the maximum and minimum value

ohe = OneHotEncoder(sparse=False)
y_one_hot = ohe.fit_transform(np.expand_dims(Y, axis=-1))
pickle.dump(ohe, open(OHE_SCALER_SAVE_PATH,'wb'))

# Split data into training and testing data
X_train,X_test,y_train,y_test = train_test_split(X_scaled, y_one_hot, 
                                                 stratify=Y, 
                                                 test_size=0.3, 
                                                 random_state=12)
Y_train = np.argmax(y_train, axis=1)
Y_test = np.argmax(y_test, axis=-1)

#%% Machine Learning 
# Using Logistic Regression because its useful to binary classification data

model = LogisticRegression()

model.fit(X_train, Y_train) 

# Model Evaluation

# Accuracy on train data
X_train_pred = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_pred,Y_train)

print('The accuracy of training data is :', train_data_accuracy)

# accuracy on test data
X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)

print('The accuracy of testing data is :', test_data_accuracy)


#%%

saved_model = 'model.pkl'
pickle.dump(saved_model, open(MODEL_PATH, 'wb'))



