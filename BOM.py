import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import openpyxl
from datetime import datetime
from sklearn import *
import copy
import numpy as np
from numpy import nan
from scipy import stats
import re
import subprocess
import time
import sys
import os
#subprocess.Popen(args=[os.getcwd()], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

def readfile():

    ## Declaring values ##
    df = pd.read_csv('uncleanedFile.csv')
    full_merged = pd.DataFrame()
    df = df.drop('Unnamed: 0',axis=1)

    ## Defining standard notation ##
    lat = re.compile('(?i).*((lat)|(Latitude))')
    long = re.compile('(?i).*((long)|(longitude))')
    HT = re.compile('(?i).*((height)|(level)|(HT)|(hgt)|(MAMSL)|(WL)|(lvl)|(metre))')
    locName = re.compile('(?i).*((locname)|(location)|(loc_name)|(LOC_NAME))')
    Date =  re.compile('(?i).*((date)|(time))')

    ## Looping to find date values ##
    for i in df.columns:
            if re.match(Date, i):
                 full_merged['Date'] = pd.to_datetime(df[str(i)])

    ## Looping to find water height values ##
    for i in df.columns:
        if re.match(HT, i):
            full_merged['HT'] = df[str(i)]

    ## Looping to find LONG values ##
    for i in df.columns:
        if re.match(long, i):
            full_merged['LONG'] = df[str(i)]


    ## Looping to find LAT values ##
    for i in df.columns:
        if re.match(lat, i):
            full_merged['LAT'] = df[str(i)]

    ## Looping to find LOC values ##
    for i in df.columns:
        if re.match(locName, i):
            full_merged['locName'] = df[str(i)]

    ## Defining main value ##
    df = full_merged
    prev = copy.copy(df)
    df.HT = df.HT.rolling(5).median()
    df.HT[0:4] = prev.HT.values[0:4]


    ## Removing outliers ##
    df[df.HT > 50] = np.nan
    df[df.HT < -50] = np.nan

    df = df.ffill().bfill()

    ## Datetime values ##
    df.Date = pd.to_datetime(df.Date)

    if 'locName' not in df.columns:
           df['locName'] = 'Single location'

    df.to_csv("Cleaneddata.csv")

readfile()
