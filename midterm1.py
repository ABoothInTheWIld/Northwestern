# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:54:05 2017

@author: t2adb
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from scipy import stats

#Define data
temperature = np.array([62, 76, 50, 51, 71,  46, 51, 44, 79])
growth = np.array([36, 39, 50, 13, 33, 33, 17, 6, 16])
df = pd.DataFrame(data= {"Temperature":temperature, "Growth":growth})

slope, intercept, r_value, p_value, std_err = stats.linregress(temperature, growth)
#print (slope, intercept, r_value, p_value, std_err)

print("The least squares line is Growth (in mm) = %.3f * Temperature (in F) + %.3f" %(slope, intercept))

predicted_temp = 52
predicted_growth = 52 * slope + intercept
print("At a temperature of %d degrees F, the predicted growth is %.3f millimeters" %(predicted_temp, predicted_growth))

#Get trend line values
trend = slope * temperature + intercept
    
#Now let's plot our data
f, ax = plt.subplots()

# Axis Labels 
plt.xlabel('Temperature (F)') 
plt.ylabel('Growth (mm)') 
plt.title('Temperature (F) vs Growth (mm)')

#scatter and regression line
plt.scatter(temperature, growth, label='Temperature vs Growth', color='b')
plt.plot(temperature, trend, label='Regression Line', color='r')

#Set legend
plt.legend(loc= 'upper right')

#Annotate r value
anchored_text = AnchoredText(" r = %.3f" %(r_value), loc=4, prop={'size':12})
ax.add_artist(anchored_text)

plt.show()