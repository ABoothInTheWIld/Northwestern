# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:41:49 2017

@author: t2adb
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:17:57 2017

@author: t2adb
"""

from scipy.optimize import linprog

#400 MidTerm
#Number 8

#Minimize 1.03x + .83y + .68z with the constraints:
#x + y + z >= 650
#.08x + .04y + .05z >= 29
#5y = 3z
#Rewrite as:
#0x + 5y - 3z = 0
#Bounds:
#x >= 0
#y >= 0
#z >= 0

#minimize equation coefficients
c = [1.03, .83, .68]

#Coefficients of constraints:
#multiply by -1 to find upper bounds
A = [[-1, -1, -1],
 [-.08, -.04, -.05]]

#Upper bounds of constraints:
B = [-650, -29]

#Equation constraints:
A_eq = [[0 , 5, -3]]
B_eq = [0]

#Boundaries
x_bounds = (0, None)
y_bounds = (0, None)
z_bounds = (0, None)

bounds = (x_bounds, y_bounds, z_bounds)

#Solve
res = linprog(c, A_ub=A, b_ub=B, A_eq= A_eq, b_eq= B_eq, bounds=bounds, options={"disp": True}, method= "simplex")

print(res)

#Solution
#x = 0, y = 243.75, z = 406.25
#minimized cost is $478.56
#Number of iterations = 3