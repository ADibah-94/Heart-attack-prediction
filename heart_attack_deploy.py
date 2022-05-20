# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:48:32 2022

@author: HP
"""

import pickle
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import streamlit as st

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') # GPU or CPU

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#%% Paths
OHE_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model','ohe_scaler.pkl')
MMS_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model','mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model','model.pkl')

#%% Loading of setting or models

scaler = MinMaxScaler()
ohe = OneHotEncoder(sparse=False)

pickle.dump(scaler, open(MMS_SCALER_PATH,'wb'))
pickle.dump(ohe, open(OHE_SCALER_PATH,'wb'))

# Machine learning 
model = 'model.pkl'
pickle.dump('saved_model', open(MODEL_PATH, 'wb'))

heart_attack_chance = {0:'negative', 1:'positive'}

#%% Test Deployment 

patient_info = np.array([41,0,1,130,204,0,0,172,0,1.4,2,0,2]).reshape(-1,1)   # true label 0
patient_info_scaled = scaler.fit_transform(patient_info)

outcome = model.predict(patient_info_scaled)
print(np.argmax(outcome))
print(heart_attack_chance[np.argmax(outcome)])

    
#%% Build your app using streamlit

with st.form('Heart Attack Preciction form'):
    st.write("Patient's info")
    age = int(st.number_input('age'))
    sex = int(st.number_input('sex'))
    cp = int(st.number_input('Chest Pain'))
    trtbps = int(st.number_input('Resting blood pressure'))
    chol = int(st.number_input('Cholestrol'))
    fbs = int(st.number_input('Fasting Blood Pressure'))
    restecg = int(st.number_input('resting electrocardiographic results'))
    thalachh = int(st.number_input('Max heart rate achieved'))
    exng = int(st.number_input('Exercise induce angina'))
    oldpeak = st.number_input('Previous peak')
    slp = int(st.number_input('Slope'))
    caa = int(st.number_input('Number of major vessel'))
    thall = int(st.number_input('Thall rate'))

    submitted = st.form_submit_button('submit')
    
    if submitted == True:
        patient_info = np.array([age, sex, cp, trtbps, chol, fbs, 
                                 restecg, thalachh, exng, oldpeak,
                                 slp, caa, thall])
        patient_info_scaled = mms.scaler.transform(np.expand_dims(patient_info,
                                                                  axis=0))
        
        outcome = model.predict(patient_info_scaled)
    
        st.write(heart_attack_chance[np.argmax(outcome)])
    
        if np.argmax(outcome)==1:
            st.warning('You going to get heart attack soon, GOOD LUCK')
        else:
            st.snow()
            st.succes('YEAH, you are free from heart attack')