# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 22:38:54 2022

@author: Dell
"""

import streamlit as st
import pickle
import numpy as np

# Loading the model
pipe = pickle.load(open('pipe8.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# To display the title
st.title('Laptop Price Predictor')

# Creating input variable

# Brand name of the laptop
Brand = st.selectbox('Brand', df['Company'].unique())

# Type of the laptop
Type = st.selectbox('Type', df['TypeName'].unique())

# RAM
RAM = st.selectbox('RAM', df['Ram'].unique())

# weight of the laptop
weight = st.number_input('Weight')

# Touchscreen
touchscreen = st.selectbox('TouchScreen', ['Yes', 'No'])

# ips
ips = st.selectbox('IPS', ['Yes', 'No'])

# ScreenResolution
ScreenResolution = st.selectbox('Screen Resolution',
                                ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
                                 '2560x1600', '2560x1440', '2304x1440'])

# Screen_size
Screen_Size = st.number_input('Screen Size')

# cpu
cpu = st.selectbox('CPU', df['Cpu_Name'].unique())

# Battery
battery = st.selectbox('Battery', df['Battery'].unique())

# HDD
hdd = st.selectbox('HDD', df['HDD'].unique())

# SSD
SSD = st.selectbox('SSD', df['SSD'].unique())

# Gpu
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())

# Os
os = st.selectbox('OS', df['OS'].unique())

if st.button("Predict Price"):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(ScreenResolution.split('x')[0])
    Y_res = int(ScreenResolution.split('x')[1])
    ppi = np.sqrt(((X_res ** 2) + (Y_res ** 2))) / Screen_Size
    val = np.array([Brand, Type, RAM, weight, touchscreen, ips, ppi, cpu, battery, hdd, SSD, gpu, os])

    val = val.reshape(1, -1)

    st.title("The predicted price for this configuration is" + str(np.exp(pipe.predict(val))))

# Apple,Ultrabook,8,1.37,No,Yes,226.9,Intel Core i5,2.3,0,256,Intel,Mac






