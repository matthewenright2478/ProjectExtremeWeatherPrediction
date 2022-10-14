## Importing Libraries ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import *
import copy



def run(select):

    ## Specifying targettted location name ##
    target = str(select)

    ## Importing libraries ##
    df = pd.read_csv('Cleaneddata.csv')
    df = df[df.locName == target]

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.resample('H').mean()
    df = df.assign(locName=target)

    prev = copy.copy(df)
    df.HT = df.HT.rolling(5).median()
    df.HT[0:4] = prev.HT.values[0:4]

    ## Removing outliers ##
    df[df.HT > 20] = np.nan
    df[df.HT < 0] = np.nan

    df = df.ffill().bfill()
    df.to_csv('FILENAME.csv')


