# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot 
from matplotlib.pyplot import *
import matplotlib.dates as mdates
import numpy as np
myFmt = mdates.DateFormatter('%m %d')


df = pd.read_csv("C:/Users/t2adb/Downloads/table.csv")

df2 = df[['Date', 'Adj Close']]
df2['Date'] = pd.to_datetime(df['Date'])    
df2['date_delta'] = (df2['Date'] - df2['Date'].min())  / np.timedelta64(1,'D')
df2 = df2.sort(['date_delta'])
df2 = df2.reset_index()

regression = pd.ols(y=df2['Adj Close'], x=df2['date_delta'])
regression.summary

trend = regression.predict()

df2['trend'] = trend

std = np.std(trend.values)

OneStd_Above = trend.values + std
OneStd_Below = trend.values - std
TwoStd_Below = OneStd_Below - std

data = pd.DataFrame(index=df2.Date, data={'Adj_Close': df2['Adj Close'].values, 'trend': trend.values,
                                          'OneStd_Above' : OneStd_Above, 'OneStd_Below' : OneStd_Below,
                                          'TwoStd_Below' : TwoStd_Below})


plt = data.plot(title = 'S&P 500, 2016')
# Shrink current axis by 20%
box = plt.get_position()
plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))