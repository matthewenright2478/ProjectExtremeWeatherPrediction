import subprocess

global hour_selected
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import pydeck as pdk
import streamlit as st
import matplotlib.pyplot as plt
from streamlit import *
import seaborn as sns
import os
import dartstest
from dartstest import darts
import calculation
from calculation import run
import BOM
from BOM import readfile
#import partial

stdin=subprocess.DEVNULL
import functools
#import partial
import subprocess

#subprocess.Popen = partial(subprocess.Popen, stdin=subprocess.DEVNULL)
## Creating title ##
st.markdown("<h1 style='text-align: center; font-size: 50px; color: red;'>Extreme Flood Prediction</h1>", unsafe_allow_html=True)

## Upload Button ##
fileName = 'Cleaneddata.csv'
files = pd.read_csv(fileName)
locations = files.locName.unique()

## Submission Form ##
with st.form("form"):
     col1, col2,col3 = st.columns([3,3,1])
     with col1:
         select = st.selectbox("Select Location",(locations),label_visibility='collapsed')

     with col2:
        model = st.selectbox('Select machine learning Model?', (
        ['ARIMA', 'BlockRNNModel', 'CatBoostModel', 'RNNModel', 'RandomForest', 'TransformerModel',
         'TCNModel', 'NHiTSModel', 'ExponentialSmoothing', 'Theta', 'FourTheta', 'Prophet',
         'RegressionModel', 'Neural Basis Expansion Analysis']), key='model',label_visibility='collapsed')

     with col3:
         submitt = st.form_submit_button("Submit")

     if submitt:
         run(select)
         darts(model)

## Creating and defining color scheme ##
named_colorscales = px.colors.named_colorscales()

## Importing forecast data  ##
df = pd.read_csv('forecast_In.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df.dropna()

## Importing forecast before data  ##
cf = pd.read_csv('forecast_Out.csv')
cf['time'] = pd.to_datetime(cf['OBS_TIME_LOC'])
cf = cf.set_index('time')
cf = cf.dropna()

## Setting up chart ##
fig = plt.figure()
sns.set_style("darkgrid")
sns.set(font_scale = 1.0,rc = {'figure.figsize':(15,8)})

## Reseting the index ##
df = df.reset_index()

## Defining max error ##
error_max = []
for i in range(0,len(df.HT)):
    error_max.append( df["HT"].values[i] * (1.2 + (i/250)))

## Defining minimum error ##
error_min = []
for i in range(0,len(df.HT)):
    if df["HT"].values[i] * (0.9 - (i/300)) > 0.001:
       error_min.append( df["HT"].values[i] * (0.9 - (i/300)))
    else:
        error_min.append(0.001)

## Setting and defining graphs ##
df['time'] = pd.to_datetime(df['time'])
projectData = pd.read_csv('projectData.csv')

plt.title(projectData.title[0])
plt.plot(df.time[0:150], df.HT[0:150], linewidth=5,color="r")
plt.fill_between(df.time[0:150], error_min[0:150], error_max[0:150], color="red", alpha=0.1)
sns.lineplot(x=cf.index[-150:-1], y=cf.HT[-150:-1],linewidth=4.0, color='b',label="Previous Data")

## Drawing the plot ##
st.pyplot(fig)
plt.show()

## Setting application dataframe widget ##
st.dataframe(df,use_container_width=True)

## Creating the upload file page ##
with st.form("Sform"):

         upload = st.file_uploader("Choose a CSV file", key='add',label_visibility='collapsed')
         submi = st.form_submit_button("Submit")
         if submi:
            dataframe = pd.read_csv(upload)
            projectData.title[0] = upload.name
            projectData.to_csv('projectData.csv')
            dataframe.to_csv('uncleanedFile.csv')
            readfile()
            fileName = 'Cleaneddata.csv'
            files = pd.read_csv(fileName)
            locations = files.locName.unique()
            select = locations[0]
            run(select)
            darts(model)