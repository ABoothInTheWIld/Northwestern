# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 22:17:18 2017

@author: Alexander
"""

#Session 1 Module 1
import numpy as np
import datetime
import time

# Exercise #1:  The volume of a sphere with radius r is (4/3)(pi)r**3.  What
# is the volume of a sphere with radius 5?  (392.6 is wrong!)

pi = 3.14159
r = 5
v = (4.0/3.0) * pi * r**3
print v

# Exercise #2:  Suppose the cover price of a book is $24.95, but bookstores get
# a 40% discount.  Shipping costs $3 for the first copy and 75 cents for each 
# additional copy.  What is the total wholesale cost for 60 copies?

copies = 60
cost = (24.95 * .6) * copies + 3 + (.75 * (copies -1))
print cost

# Exercise #3:  If I leave my house at 6:52 am and run 1 mile at an easy pace
# (8:15 per mile), then 3 miles at tempo (7:12 per mile) and 1 mile at easy
# pace again, what time do I get home for breakfast?

#init datetime with arbitrary date but specific time
#Year, Month, Day, Hour, Minute
d = datetime.datetime(2017, 01, 03, 6, 52)

#init easy and tempo pace timedeltas
easyPace = datetime.timedelta(minutes = 8, seconds = 15)
tempPace = datetime.timedelta(minutes = 7, seconds = 12)

#Calculate breakfast time
breakfast = d + 2 * easyPace + 3 * tempPace
print breakfast.time()

#By using an arbitrary date, we can create a full datetime.datetime object and
#use operators with timedelta upon on it
#Code is in Python 2.7.11
