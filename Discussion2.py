# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:33:12 2017

@author: Alexander
"""

import csv
import os

#set wd
os.chdir('C:/Users/Alexander/Documents/Northwestern/Winter 2017/MSPA 400/Python/Session 2')

#open csvs
trainData = open('train.csv', "r").readlines()
trainLabels = open('trainLabels.csv', "r").readlines()

#init lists
trainDataList = []
trainLabelList = []

#strip out the new line character
for line in trainData:      
    actual_line = line.rstrip('\n')
    trainDataList.append(actual_line)

for line in trainLabels:      
    actual_line = line.rstrip('\n')
    trainLabelList.append(actual_line)

#Open the combined csv to write too
i = 0
with open('trainCombined.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        #loop through data rows
        for row in trainDataList:
            #Add label to the beginning
            row = trainLabelList[i] + ',' + row
            #split on commas and write to cells in a row
            writer.writerow([c.strip() for c in row.strip(', ').split(',')])
            i = i + 1
        
