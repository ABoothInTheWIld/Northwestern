# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:43:43 2017

@author: Alexander
"""

# Imports
# prepare for Python version 3x features and functions
from __future__ import division, print_function

import os
import pandas as pd      
import numpy as np  # arrays and math functions
import seaborn as sns
import matplotlib.pyplot as plt
from re import sub
from decimal import Decimal
import statsmodels.formula.api as smf  # R-like model specification
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn import metrics
import scipy.stats as stats

#Set Directory
os.chdir(r"C:\Users\Alexander\Documents\Northwestern\Fall 2017\411\Unit 02\Project 2")

#Read in the auto ins dataset
train = pd.read_csv('train_auto.csv')
test = pd.read_csv('test_auto.csv')

#A good step to take is to convert all variable names to lower case
train.columns = [s.lower() for s in train.columns]
test.columns = [s.lower() for s in train.columns]

#get columns for reference
cols = list(train.columns.values)
cols.sort()

print('')
print('----- Summary of Input Data -----')
print('')

# show the object is a DataFrame
print('Object type: ', type(train))

# show the object's shape
print('Object shape: ', train.shape)

# show number of observations in the DataFrame
print('Number of observations: ', len(train))

# show variable names
print('Variable names: ', train.columns)

# show head
print(train.head(5))

# show descriptive statistics
print(train.describe())

#Na Analysis
naColsCount = train.isnull().sum()
naCols = train.isnull().any()

#Note the missing values are recorded as NaN, we need to replace these with something 
#(median? or your choice)
train1 = train.copy()

#One of the columns with NaN is categorical. Replace with "No Job Info"
train1.job = train.job.fillna("No Job Info")
naCols = train1.isnull().any()

#there are dollar signs and commas in some of these variables
#convert to int
for indx in cols:
    if train[indx].astype(str).str.contains("\$").any() == True:
        tempVals = []
        for money in train[indx].values:
            if not pd.isnull(money):
                value = int(Decimal(sub(r'[^\d\-.]', '', money)))
                tempVals.append(value)
            else:
                tempVals.append(np.nan)
        train1[indx] = tempVals

#Create home own variable. 1 if home val is greater than 0 and not NaN
train1["home_own"] = np.where(train1["home_val"].fillna(0) > 0, 1, 0)

#start by converting all NaN values to 0
train1=train1.fillna(0)

#convert all NaN values to median
for indx in cols:
    if naCols[indx] == True:
        newName = "IMP_" + indx
        newInd = "m_" + indx
        m = np.median(train1[indx][train1[indx] > 0])
        train1[newName] = train1[indx].replace({0: m}).astype(int)
        train1[newInd] = (train1[indx] == 0).astype(int)

#delete variables that had missing variable imputed
df = train1.copy()
del(df["index"])
del(df["yoj"])
del(df["income"])
del(df["home_val"])
del(df["age"])
del(df["car_age"])
##################################################################################
dfCols = df.columns.values

#Gets SIGNIFICANT outliers
def get_outliers(pdSer):
    """Get outliers from a pandas series
    """
    quants = pdSer.quantile([.25, .75])
    iqr = quants[.75] - quants[.25]
    
    lower = quants[.25] - 3 * iqr
    upper = quants[.75] + 3 * iqr
    return [value for value in pdSer if value < lower or value > upper]


#EDA
print(df.describe())

contVars = ['target_amt', 'target_flag', 'travtime', 'bluebook', 'tif', 'oldclaim', 
            'clm_freq', 'mvr_pts', 'IMP_age', 'IMP_car_age','IMP_home_val',
            'IMP_income', 'IMP_yoj']
#EDA
#Targets
#counts of Flag
df.target_flag.value_counts()

#Plots of Amnt
ax = sns.boxplot(df[df.target_amt > 0].target_amt, orient='v')
ax.set_title("Boxplot of Target Amount")
ax.set_ylabel("Amount")
plt.show()

ax = sns.distplot(df[df.target_amt > 0].target_amt, kde=False)
ax.set_title("Histogram of Target Amount")
plt.show()

#Get outliers
print(get_outliers(df.target_amt))
#Couple of outliers here

#Categorical
#counts of kidsdriv
df.kidsdriv.value_counts()

#counts of homekids
df.homekids.value_counts()

#counts of parent1
df.parent1.value_counts()

#counts of mstatus
df.mstatus.value_counts()

#counts of sex
df.sex.value_counts()

#counts of education
df.education.value_counts()
#need to fix high School
df.loc[df.education == "<High School", 'education'] = "z_High School"
df.education.value_counts()

#counts of job
df.job.value_counts()

#counts of car_use
df.car_use.value_counts()

#counts of car_type
df.car_type.value_counts()

#counts of red_car
df.red_car.value_counts()

#counts of revoked
df.revoked.value_counts()

#counts of urbanicity
df.urbanicity.value_counts()

#counts of clm_freq
df.clm_freq.value_counts()

df["white_collar"] = np.where(df["job"].isin(["Doctor", "Manager", 
       "Lawyer"]), 1, 0)
df["university_degree"] = np.where(df["education"].isin(["Bachelors",
       "Masters", "PhD"]), 1, 0)

df.white_collar.value_counts()

df.university_degree.value_counts()

df.home_own.value_counts()

#Continuous
#Plots of travtime
ax = sns.boxplot(df.travtime, orient='v')
ax.set_title("Boxplot of travtime")
ax.set_ylabel("travtime")
plt.show()

ax = sns.distplot(df.travtime, kde=False)
ax.set_title("Histogram of travtime")
plt.show()

#Get outliers
print(get_outliers(df.travtime))
#couple of outliers

#Plots of bluebook
ax = sns.boxplot(df.bluebook, orient='v')
ax.set_title("Boxplot of bluebook")
ax.set_ylabel("bluebook")
plt.show()

ax = sns.distplot(df.bluebook, kde=False)
ax.set_title("Histogram of bluebook")
plt.show()

#Get outliers
print(get_outliers(df.bluebook))
#couple of outliers

#Plots of tif
ax = sns.boxplot(df.tif, orient='v')
ax.set_title("Boxplot of tif")
ax.set_ylabel("tif")
plt.show()

ax = sns.distplot(df.tif, kde=False)
ax.set_title("Histogram of tif")
plt.show()

#Get outliers
print(get_outliers(df.tif))

#Plots of oldclaim
ax = sns.boxplot(df[df.oldclaim > 0].oldclaim, orient='v')
ax.set_title("Boxplot of oldclaim")
ax.set_ylabel("oldclaim")
plt.show()

ax = sns.distplot(df[df.oldclaim > 0].oldclaim, kde=False)
ax.set_title("Histogram of oldclaim")
plt.show()

#Get outliers
print(get_outliers(df[df.oldclaim > 0].oldclaim))

df["hadOldClaim"] = np.where(df["oldclaim"] > 0, 1, 0)
df["hadFewOldClaim"] = np.where((df["oldclaim"] > 0) & (df["oldclaim"] <= 12000), 1, 0)
df["hadManyOldClaim"] = np.where(df["oldclaim"] > 12000, 1, 0)

df.hadOldClaim.value_counts()
df.hadFewOldClaim.value_counts()
df.hadManyOldClaim.value_counts()

#Plots of clm_freq
ax = sns.distplot(df.clm_freq, kde=False)
ax.set_title("Histogram of clm_freq")
plt.show()


#Plots of mvr_pts
ax = sns.boxplot(df.mvr_pts, orient='v')
ax.set_title("Boxplot of mvr_pts")
ax.set_ylabel("mvr_pts")
plt.show()

ax = sns.distplot(df.mvr_pts, kde=False)
ax.set_title("Histogram of mvr_pts")
plt.show()

#Get outliers
print(get_outliers(df.mvr_pts))

#Plots of IMP_age
ax = sns.boxplot(df.IMP_age, orient='v')
ax.set_title("Boxplot of IMP_age")
ax.set_ylabel("IMP_age")
plt.show()

ax = sns.distplot(df.IMP_age, kde=False)
ax.set_title("Histogram of IMP_age")
plt.show()

#Get outliers
print(get_outliers(df.IMP_age))

#Plots of IMP_car_age
ax = sns.boxplot(df.IMP_car_age, orient='v')
ax.set_title("Boxplot of IMP_car_age")
ax.set_ylabel("IMP_car_age")
plt.show()

ax = sns.distplot(df.IMP_car_age, kde=False)
ax.set_title("Histogram of IMP_car_age")
plt.show()

#Get outliers
print(get_outliers(df.IMP_car_age))

#Plots of IMP_home_val
ax = sns.boxplot(df.IMP_home_val, orient='v')
ax.set_title("Boxplot of IMP_home_val")
ax.set_ylabel("IMP_home_val")
plt.show()

ax = sns.distplot(df.IMP_home_val, kde=False)
ax.set_title("Histogram of IMP_home_val")
plt.show()

#Get outliers
print(get_outliers(df.IMP_home_val))
#Fix outliers

#Plots of IMP_income
ax = sns.boxplot(df.IMP_income, orient='v')
ax.set_title("Boxplot of IMP_income")
ax.set_ylabel("IMP_income")
plt.show()

ax = sns.distplot(df.IMP_income, kde=False)
ax.set_title("IMP_income")
plt.show()

#Get outliers
print(get_outliers(df.IMP_income))
#fix outliers

#Plots of IMP_yoj
ax = sns.boxplot(df.IMP_yoj, orient='v')
ax.set_title("Boxplot of IMP_yoj")
ax.set_ylabel("IMP_yoj")
plt.show()

ax = sns.distplot(df.IMP_yoj, kde=False)
ax.set_title("IMP_yoj")
plt.show()

#Get outliers
print(get_outliers(df.IMP_yoj))

##########################################################
def fixOutliers(onePdSer, otherPdSer, q1, q2):
    newpdSer = onePdSer.copy()
    for val in range(0, len(onePdSer)):
        if onePdSer[val] > otherPdSer.quantile(q1):
            newpdSer[val] = otherPdSer.quantile(q1)
        
        if onePdSer[val] < otherPdSer.quantile(q2):
            newpdSer[val] = otherPdSer.quantile(q2)
    
    return newpdSer

#truncate outliers
df.target_amt = fixOutliers(df.target_amt, df.target_amt, .99, .01)
df.IMP_income = fixOutliers(df.IMP_income, df.IMP_income, .99, .01)
df.IMP_home_val = fixOutliers(df.IMP_home_val, df.IMP_home_val, .99, .01)
df.bluebook = fixOutliers(df.bluebook, df.bluebook, .99, .01)
df.travtime = fixOutliers(df.travtime, df.travtime, .99, .01)

#Get dummies
cols_to_dum = [ 'IMP_red_car', 'IMP_urbanicity', 'IMP_revoked',
                 'IMP_sex','IMP_mstatus','IMP_car_use', 'IMP_parent1',
                 "home_own", "university_degree", "white_collar",
             'kidsdriv', 'homekids', "hadOldClaim", "hadFewOldClaim", "hadManyOldClaim"]

df["IMP_mstatus"] = np.where(df["mstatus"].isin(["Yes"]), 1, 0)
df["IMP_parent1"] = np.where(df["parent1"].isin(["Yes"]), 1, 0)
df["IMP_sex"] = np.where(df["sex"].isin(["z_F"]), 1, 0)
df["IMP_revoked"] = np.where(df["revoked"].isin(["Yes"]), 1, 0)
df["IMP_red_car"] = np.where(df["red_car"].isin(["yes"]), 1, 0)
df["IMP_urbanicity"] = np.where(df["urbanicity"].isin(["Highly Urban/ Urban"]), 1, 0)
df["IMP_car_use"] = np.where(df["car_use"].isin(["Commercial"]), 1, 0)

#continuous vars correlation
df[contVars].corrwith(df.target_flag)

#categorical vars correlation
df[cols_to_dum].corrwith(df.target_flag)

#only drop yoj and red_car due to essentially 0 correlation
potentCols = ["target_flag", "target_amt",
 'bluebook',
 'kidsdriv',
 'homekids',
 'oldclaim',
 'clm_freq',
 'mvr_pts',
 'tif',
 'travtime',
 'IMP_sex',
 'IMP_age',
 'IMP_car_age',
 'IMP_home_val',
 'IMP_income', 'IMP_urbanicity', 'IMP_revoked',
 'IMP_mstatus','IMP_car_use', 'IMP_parent1',
 "home_own", "university_degree", "white_collar", "car_type",
 "hadOldClaim", "hadFewOldClaim", "hadManyOldClaim"]

df_sub = df[potentCols]

#Collinearity Check
corrs = df_sub.corr()
print(corrs)

#Get heatmap
ax = plt.axes()
sns.heatmap(corrs, 
            xticklabels=corrs.columns.values,
            yticklabels=corrs.columns.values, ax = ax)

ax.set_title('Insurance Correlation Heatmap')
plt.show()

#sns.pairplot(df_sub)

####################################################
features = "+".join(df_sub.columns[2:27])
logit1 = smf.logit('target_flag ~ ' + features, data=df_sub).fit()
logit1.summary()
logit1.summary2()

features2 =  "clm_freq+mvr_pts"
logit12 = smf.logit('target_flag ~ ' + features2, data=df_sub).fit()
logit12.summary()
logit12.summary2()

features3 =  "clm_freq+mvr_pts+IMP_urbanicity"
logit3 = smf.logit('target_flag ~ ' + features3, data=df_sub).fit()
logit3.summary()
logit3.summary2()

######REGRESSION ONE######################
features4 =  "clm_freq+mvr_pts+IMP_urbanicity+IMP_revoked+IMP_parent1"
logit4 = smf.logit('target_flag ~ ' + features4, data=df_sub).fit()
logit4.summary()
logit4.summary2()

preds = logit4.predict()
fpr, tpr, _ = metrics.roc_curve(df_sub['target_flag'], preds)
 
# calculate AUC and create ROC curve
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

############################################################
#Gets the columns without collinearity
def getBestCols_LOGIT(inpDF, inpYString, thresh = 3):
    tempList = list(inpDF.columns.values)
    tempList.remove(inpYString)
    inpFeatures = "+".join(tempList)
    y, X = dmatrices(inpYString + " ~ " + inpFeatures, inpDF, return_type='dataframe')
    # For each X, calculate VIF and save in dataframe
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["Features"] = X.columns
    #check if collinearity is below thresh limit
    isDone = all(n < thresh for n in vif[vif.Features != "Intercept"]["VIF Factor"])
    if isDone:
        return vif.Features
    else:
        #Get the Features with the top 2 VIF
        vif_sub = vif[vif.Features != "Intercept"]
        colToTest1 = vif_sub[vif_sub["VIF Factor"] == max(vif_sub["VIF Factor"])]["Features"].values[0]
        vif_sub2 = vif_sub[vif_sub.Features != colToTest1]
        colToTest2 = vif_sub2[vif_sub2["VIF Factor"] == max(vif_sub2["VIF Factor"])]["Features"].values[0]
        
        #Drop one and create model
        testDF1 = inpDF.drop(colToTest1, 1)
        tempList1 = list(testDF1.columns.values)
        tempList1.remove(inpYString)
        testFeatures1 = "+".join(tempList1)
        firstModel = smf.logit(inpYString + " ~ " + testFeatures1, data=testDF1).fit()
        
        #Drop other and create model
        testDF2 = inpDF.drop(colToTest2, 1)
        tempList2 = list(testDF2.columns.values)
        tempList2.remove(inpYString)
        testFeatures2 = "+".join(tempList2)
        secondModel = smf.logit(inpYString + " ~ " + testFeatures2, data=testDF2).fit()
        
        #Pick better model and recurse
        if firstModel.prsquared > secondModel.prsquared:
            return getBestCols_LOGIT(testDF1, inpYString)
        elif secondModel.prsquared > firstModel.prsquared:
            return getBestCols_LOGIT(testDF2, inpYString)
        else:
            return getBestCols_LOGIT(testDF1, inpYString)

##################################
df_sub2 = df_sub.copy()
del(df_sub2["target_amt"])
del(df_sub2["car_type"])
 
theBestCols = getBestCols_LOGIT(df_sub2, "target_flag").values
index = np.argwhere(theBestCols=="Intercept")
theBestCols = np.delete(theBestCols, index)
theBestCols = np.append(theBestCols, "target_flag")
df_sub_VIF = df_sub2[theBestCols]

#Double check the results of the above function
tempList = list(df_sub_VIF.columns.values)
tempList.remove("target_flag")
testFeatures = "+".join(tempList)
y, X = dmatrices('target_flag ~' + testFeatures, df_sub_VIF, return_type='dataframe')
# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Features"] = X.columns

print(vif)

#Looks good!
##################################
df_sub_VIF["car_type"] = df_sub["car_type"]
tempList = list(df_sub_VIF.columns.values)
tempList.remove("target_flag")
testFeatures = "+".join(tempList)

###########REFGRESSION 2############################333#
logit5 = smf.logit('target_flag ~ ' + testFeatures, data=df_sub_VIF).fit()
logit5.summary()
logit5.summary2()

preds = logit5.predict()
fpr, tpr, _ = metrics.roc_curve(df_sub['target_flag'], preds)
 
# calculate AUC and create ROC curve
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

logit5.params

###########################REGRESSION 3#########################3
df_sub3 = df_sub_VIF.copy()
del(df_sub3["IMP_sex"])
del(df_sub3["IMP_age"])
del(df_sub3["IMP_car_age"])
del(df_sub3["IMP_income"])
del(df_sub3["homekids"])
del(df_sub3["hadManyOldClaim"])

tempList = list(df_sub3.columns.values)
tempList.remove("target_flag")
testFeatures = "+".join(tempList)
logit6 = smf.logit('target_flag ~ ' + testFeatures, data=df_sub3).fit()
logit6.summary()
logit6.summary2()

logit6.params

preds = logit6.predict()
fpr, tpr, _ = metrics.roc_curve(df_sub['target_flag'], preds)
 
# calculate AUC and create ROC curve
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

#The coefficients for bluebook, home_val, and oldclaim are tiny

#############################REGRESSION 4###############
df_sub3["log_bluebook"] = np.log(df_sub3.bluebook)
df_sub3["log_IMP_home_val"] = np.log(df_sub3.IMP_home_val)
#df_sub3["log_IMP_income"] = np.log(df_sub3.IMP_income)

tempList = list(df_sub3.columns.values)
tempList.remove("target_flag")
tempList.remove("bluebook")
tempList.remove("IMP_home_val")

testFeatures = "+".join(tempList)
logit7 = smf.logit('target_flag ~ ' + testFeatures, data=df_sub3).fit()
logit7.summary()
logit7.summary2()

preds = logit7.predict()

fpr, tpr, _ = metrics.roc_curve(df_sub3['target_flag'], preds)
 
# calculate AUC and create ROC curve
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

#Finally for the Hosmer-Lemeshow Lack of Fit
pred_train = preds
pred_out = df_sub3.loc[:,['index','target_flag']]
pred_out['p_target_flag'] = pred_train[: ]
pred_out.head 
#sort by pred
result1 = pred_out.sort_values(by=('p_target_flag'))
print (result1)

#rank by decile
result2 = pd.qcut(result1['p_target_flag'], 10, labels=[1,2,3,4,5,6,7,8,9,10])
print (result2)
d1 = {'g' : result2}
df3 = pd.DataFrame(data=d1)
print (df3)
result3 = pd.concat([result1, df3], axis=1, join_axes=[result1.index])
print (result3)

sums = result3.groupby('g')
sums1 = sums.aggregate(np.sum)
print (sums1)

#plot target_flag vs pred
def scat(dataframe,var1,var2,var3):     
    dataframe[var2].plot()     
    dataframe[var3].plot()     
    plt.title('Hosmer-Lemeshow lack of fit trng data')     
    plt.xlabel(var1)     
    plt.ylabel('Sum by Group')
    
scat(sums1, 'g', 'target_flag', 'p_target_flag')

#########################################################

df2 = df[df.target_flag == 1]

#continuous vars correlation
df2[contVars].corrwith(df2.target_amt)

#categorical vars correlation
df2[cols_to_dum].corrwith(df2.target_amt)

df2["log_bluebook"] = np.log(df2.bluebook)

features2 = "log_bluebook+IMP_mstatus+mvr_pts"
result2 = smf.ols('target_amt ~' + features2, data=df2).fit()
print(result2.summary())

y = df2.target_amt

#Get predictions and residuals
x2_predictions = result2.predict()
x2_residuals = y - x2_predictions

#Plots
#Residuals vs Fitted
fig = plt.figure()
plt.scatter(x=x2_predictions, y= x2_residuals)
fig.suptitle('Cost Model Residuals vs Fitted Target Cost')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

#QQPlot
stats.probplot(x2_residuals, dist="norm", plot=plt)
plt.show()

print(metrics.mean_absolute_error(y, x2_predictions))


###########################################################################
#Do all of this to the test set
test1 = test.copy()
#del(test1["target_amt"])
#del(test1["target_flag"])
testCols = list(test1.columns.values)

#One of the columns with NaN is categorical. Replace with "No Job Info"
test1.job = test1.job.fillna("No Job Info")
naColsTest = test1.isnull().any()

#there are dollar signs and commas in some of these variables
#convert to int
for indx in testCols:
    if test1[indx].astype(str).str.contains("\$").any() == True:
        tempVals = []
        for money in test1[indx].values:
            if not pd.isnull(money):
                value = int(Decimal(sub(r'[^\d\-.]', '', money)))
                tempVals.append(value)
            else:
                tempVals.append(np.nan)
        test1[indx] = tempVals

#Create home own variable. 1 if home val is greater than 0 and not NaN
test1["home_own"] = np.where(test1["home_val"].fillna(0) > 0, 1, 0)

#start by converting all NaN values to 0
test1=test1.fillna(0)

#convert all NaN values to median
for indx in testCols:
    if naColsTest[indx] == True:
        newName = "IMP_" + indx
        newInd = "m_" + indx
        m = np.median(train1[indx][train1[indx] > 0])
        test1[newName] = test1[indx].replace({0: m}).astype(int)
        test1[newInd] = (test1[indx] == 0).astype(int)

#need to fix high School
test1.loc[test1.education == "<High School", 'education'] = "z_High School"
test1.education.value_counts()

#truncate outliers
test1.IMP_income = fixOutliers(test1.IMP_income, train1.IMP_income, .99, .01)
test1.IMP_home_val = fixOutliers(test1.IMP_home_val, train1.IMP_home_val, .99, .01)
test1.bluebook = fixOutliers(test1.bluebook, df.bluebook, .99, .01)
test1.travtime = fixOutliers(test1.travtime, df.travtime, .99, .01)

test1["white_collar"] = np.where(test1["job"].isin(["Doctor", "Manager", 
       "Lawyer"]), 1, 0)
test1["university_degree"] = np.where(test1["education"].isin(["Bachelors",
       "Masters", "PhD"]), 1, 0)

#Get new variables
test1["hadOldClaim"] = np.where(test1["oldclaim"] > 0, 1, 0)
test1["hadFewOldClaim"] = np.where((test1["oldclaim"] > 0) & (test1["oldclaim"] <= 12000), 1, 0)
test1["hadManyOldClaim"] = np.where(test1["oldclaim"] > 12000, 1, 0)

test1["IMP_mstatus"] = np.where(test1["mstatus"].isin(["Yes"]), 1, 0)
test1["IMP_parent1"] = np.where(test1["parent1"].isin(["Yes"]), 1, 0)
test1["IMP_sex"] = np.where(test1["sex"].isin(["z_F"]), 1, 0)
test1["IMP_revoked"] = np.where(test1["revoked"].isin(["Yes"]), 1, 0)
test1["IMP_red_car"] = np.where(test1["red_car"].isin(["yes"]), 1, 0)
test1["IMP_urbanicity"] = np.where(test1["urbanicity"].isin(["Highly Urban/ Urban"]), 1, 0)
test1["IMP_car_use"] = np.where(test1["car_use"].isin(["Commercial"]), 1, 0)

test1["log_bluebook"] = np.log(test1.bluebook)
test1["log_IMP_home_val"] = np.log(test1.IMP_home_val)

#use this model to predict for the test data
preds = logit7.predict(test1)
preds.head
pred_out = test1.loc[:,['index','target_flag']]
pred_out['p_target_flag'] = preds[: ]
pred_out.head

preds2 = result2.predict(test1)
pred_out['p_target_amt'] = preds2[: ]
pred_out.head
#watch your record count should be 2141
your_model = pred_out.loc[:,['index','p_target_flag','p_target_amt']]  
your_model.head 
your_model.to_csv('booth_alexander_hw02_predictions_final.csv')