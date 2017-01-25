# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:39:02 2017

@author: Alexander Booth
"""

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats
import numpy as np

#Graphing Funtion
def graphNump(arr1, arr2):
    
    #Test to see if we can graph the arrays
    if not is_graphable(arr1, arr2):
        return
    
    #Get regression stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(arr1,arr2)
    #print (slope, intercept, r_value, p_value, std_err)
    
    #Get trend line values
    trend = slope * arr1 + intercept
    
    # Now lets plot our data
    f, ax = plt.subplots()
    
    # Axis Labels 
    plt.xlabel('Array 1') 
    plt.ylabel('Array 2') 
    plt.title('Array 1 vs Array 2')
    
    #scatter and regression line
    plt.scatter(arr1, arr2, label='Array1 vs Array2', color='b')
    plt.plot(arr1, trend, label='Regression Line', color='r')
    
    #Set legend
    plt.legend(loc= 'upper left')
    
    #Annotate r value
    anchored_text = AnchoredText(" r = %.3f" %(r_value), loc=4, prop={'size':12})
    ax.add_artist(anchored_text)
    
    plt.show()
    return

#Check if 2 arrays are graphable
def is_graphable(arr1, arr2):
    
    #check shape
    if arr1.shape != arr2.shape:
        print("These arrays are different shapes!")
        return False
    #Check dimensions
    elif len(list(arr1.shape)) != 1 or arr1.dtype == 'O':
        print("The first array has more than one dimension!")
        return False
    #Check arithmetic operations can be performed
    elif not is_numeric(arr1):
        print("The first array seems not to be numeric...")
        return False
    #Repeat for second array
    elif len(list(arr2.shape)) != 1 or arr2.dtype == 'O':
        print("The second array has more than one dimension!")
        return False
    elif not is_numeric(arr2):
        print("The second array seems not to be numeric...")
        return False
    else:
        print("Nice, we can graph these arrays")
        return True

#Check arithmetic operators on an object        
def is_numeric(obj):
    try:
        obj+obj, obj-obj, obj*obj, obj**obj, obj/obj
    except ZeroDivisionError:
        return True
    except Exception:
        return False
    else:
        return True

#Fails Shape
arr1 = np.array([1,2,3, 4])
arr2 = np.array([1,2,3])
graphNump(arr1, arr2)

#Fails Dimension
arr1 = np.array([[1,2,3],[1,2,3]])
arr2 = np.array([[1,2,3],[1,2,3]])
graphNump(arr1, arr2)

#Fails Shape & Dimension
arr1 = np.array([[1,2,3],[1,2]])
arr2 = np.array([[1,2,3],[1,2]])
graphNump(arr1, arr2)

#Fails Numeric
arr1 = np.array(['String', 5, 7])
arr2 = np.array([1,2,13])
graphNump(arr1, arr2)

#Plot 2 random arrays
arr1 = np.random.rand(1, 10)[0] * 100
arr2 = np.random.rand(1, 10)[0] * 100
graphNump(arr1, arr2)