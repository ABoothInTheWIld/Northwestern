# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:51:06 2017

@author: t2adb
"""

import numpy as np
from scipy.optimize import linprog

upper_bounds = np.array([
    [1, 2],  # Fabrication limitations
    [3, 4]   # Finishing limitations 
])

ub_constraints_row = np.array([80, 180])  # Fab and Finish caps

# Moved the obj values to the left side of the equation to match the upper bounds above
objective = np.array([-130, -160])  

x0_bounds = (0.0, None)
x1_bounds = (0.0, None)

result = linprog(c=objective,
                 A_ub=upper_bounds,
                 b_ub=ub_constraints_row,
                 bounds=(x0_bounds,
                         x1_bounds
                         )
                )

# The optimal value is negative because we moved the obj values. So make it positive
print ("Optimal Value: %s" % (result.fun * -1))
print ("Parameters at Optimal Value: %s" % result.x)