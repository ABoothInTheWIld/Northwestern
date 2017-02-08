# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 19:11:07 2017

@author: Alexander
"""

import random
import math
import matplotlib.pyplot 
from matplotlib.pyplot import *

def f(x):
    # A function to generate example y values
    r = random.random()
    y = x-10*r-x**r
    return y

random.seed(352) # Ensures that we all generate the same random numbers

x_values = range(1,101) # Numbers from 1 to 100

y_values = [f(x) for x in x_values] # Get y value for each x value

n = len(x_values) # the number of observations

#my code
#define variables for best fit line and correlation coefficient
sigXY = sum([x*y for x,y in zip(x_values, y_values)])
sigX = sum(x_values)
sigY = sum(y_values)
sigX2 = sum([x*x for x in x_values])
sigY2 = sum([y*y for y in y_values])

#define slope
slopeNumer = n * sigXY - sigX*sigY
slopeDenom = n * sigX2 - sigX **2.0

#define best fit line
m = slopeNumer/slopeDenom
b = (sigY - m*sigX)/n
linReg_yValues = [m*x + b for x in x_values]

rNumer = (n * sigXY) - (sigX * sigY)
rDenom = (math.sqrt(n * sigX2 - sigX**2)) * (math.sqrt(n* sigY2 - sigY**2))

r = rNumer/rDenom

figure()
plot(x_values, y_values, 'b')
plot(x_values, linReg_yValues, 'r')
legend(('Random Seed','Line of Best Fit'),loc=4) 
title('Python Discussion 2') 
show()

print "Line of Best Fit equals %f*x + %f" %(m,b)
print "r = %f" %r
