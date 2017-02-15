# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:31:11 2017

@author: Alexander
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:17:57 2017

@author: t2adb
"""

from scipy.optimize import linprog

#Maximize x + y with the constraints:

#Bounds:
#x >= 0
#x <= 5
#y >= 0
#y <= 5

#maximize equation coefficients
#max(f(x)) == -min(-f(x))
#so multiply f(x) by -1

c = [-1, -1]

#Boundaries
x_bounds = (0, 5)
y_bounds = (0, 5)

bounds = (x_bounds, y_bounds)

#Solve
res = linprog(c, bounds=bounds, options={"disp": True}, method= "simplex")

print(res)

#max(f(x)) == -min(-f(x)):
#Multiply -10 by -1 to get max

#Solution
#x = 5, y = 5
#maximum = 10
#Number of iterations = 2