# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:17:57 2017

@author: t2adb
"""

from scipy.optimize import linprog

#400 MidTerm
#Number 6

#Maximize 528x + 492y + 348z with the constraints:
#x + y + z <= 12
#z >= 3y
#Rewrite as:
#0x + 3y - z <= 0
#Bounds:
#x >= 0
#x <= 4
#y >= 0
#z >= 0

#maximize equation coefficients
#max(f(x)) == -min(-f(x))
#so multiply f(x) by -1

c = [-528, -492, -348]

#Coefficients of contstraints:
A = [[1, 1, 1],
 [0, 3, -1]]

#Upper bounds of constraints:
B = [12, 0]

#Boundaries
x_bounds = (0, 4)
y_bounds = (0, None)
z_bounds = (0, None)

bounds = (x_bounds, y_bounds, z_bounds)

#Solve
res = linprog(c, A_ub=A, b_ub=B, bounds=bounds, options={"disp": True}, method= "simplex")

print(res)

#max(f(x)) == -min(-f(x)):
#Multiply -5184 by -1 to get max

#Solution
#x = 4, y = 2, z = 6
#maximized calories = 5,184
#Number of iterations = 3