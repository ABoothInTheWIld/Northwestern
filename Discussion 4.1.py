# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 21:12:38 2017

@author: Alexander
"""

from scipy.optimize import linprog

#Exercise 4.3.27
#Page 180

#Minimize x + y + z with the constraints:
#3.5x + 4y + 8z >= 1500
#x + y >= 3z
#rewrite as:
#x + y -3z >= 0
#Bounds:
#x >= 30
#y >= 0
#z >= 0

#minimze equation coefficients
c = [1, 1, 1]

#Since constraints are greater than, multiple all by -1 to create upper bounds
A = [[-3.5, -4, -8],
      [-1, -1, 3]]

#Coefficients of constraints
B = [-1500, 0]

#Boundaries
x_bounds = (30, None)
y_bounds = (0, None)
z_bounds = (0, None)

bounds = (x_bounds, y_bounds, z_bounds)

res = linprog(c, A_ub=A, b_ub=B, bounds=bounds, options={"disp": True})

print(res)

#x = 30, y = 197.25, z = 75.75
#Minimized sum = 303.0
#Number of iterations = 3
