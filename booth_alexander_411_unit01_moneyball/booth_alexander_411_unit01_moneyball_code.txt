# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 00:35:44 2017

@author: Alexander
"""

# prepare for Python version 3x features and functions
from __future__ import division, print_function

#Imports
import os
import math
import pandas as pd
from pandas.tools.plotting import scatter_matrix 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import abline_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import feature_selection
from sklearn.linear_model import LinearRegression

#Set Directory
os.chdir(r"C:\Users\Alexander\Documents\Northwestern\Fall 2017\411\Unit 01\booth_alexander_411_unit01_moneyball")

#Load the training data and the testing data
train = pd.read_csv(r'moneyball_train.csv')
test = pd.read_csv(r'moneyball_test.csv')
 
#get columns for reference
cols = list(train.columns.values)
cols.sort()

naColsCount = train.isnull().sum()
naCols = train.isnull().any()

#Note the missing values are recorded as NaN, we need to replace these with something 
#(median? or your choice)
#start by converting all NaN values to 
train1 = train.copy(deep=False)

for indx in cols:
    if naCols[indx] == True:
        newName = "IMP_" + indx
        newInd = "m_" + indx
        m = np.median(train1[indx][train1[indx] > 0])
        train1[newName] = train[indx].fillna(m)
        train1[newInd] = train[indx].isnull().astype(int)

print(train1[train1.values == 0].count())

train1["TEAM_BATTING_OBC"] = train1.TEAM_BATTING_H + train1.TEAM_BATTING_BB + train1.IMP_TEAM_BATTING_HBP 
train1["TEAM_BATTING_1B"] = train1.TEAM_BATTING_H - train1.TEAM_BATTING_2B - train1.TEAM_BATTING_3B - train1.TEAM_BATTING_HR
train1["TEAM_BATTING_TOTALBASES"] = train1.TEAM_BATTING_1B + 2*train1.TEAM_BATTING_2B + 3*train1.TEAM_BATTING_3B + 4*train1.TEAM_BATTING_HR
train1["TEAM_PITCHING_OBC"] = train1.TEAM_PITCHING_BB + train1.TEAM_PITCHING_H

#A good step to take is to convert all variable names to lower case
train1.columns = [s.lower() for s in train1.columns]

df = train1.copy(deep=False)
del(df["index"])
del(df["unnamed: 0"])
del(df["team_batting_so"])
del(df["team_baserun_sb"])
del(df["team_baserun_cs"])
del(df["team_batting_hbp"])
del(df["team_fielding_dp"])
del(df["team_pitching_so"])

########################################################################33
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

#EDA
#Plots of Wins
ax = sns.boxplot([df.target_wins], orient='v')
ax.set_title("Boxplot of Target Wins")
ax.set_ylabel("Wins")
plt.show()

ax = sns.distplot(df.target_wins, kde=False)
ax.set_title("Histogram of Wins")
plt.show()

#Get outliers
print(get_outliers(df.target_wins))


#There is a point with zero wins. Probably an outlier

################################################
#Batting
#Plots of batting H
ax = sns.boxplot([df.team_batting_h], orient='v')
ax.set_title("Boxplot of Team Batting Hits")
ax.set_ylabel("Hits")
plt.show()

ax = sns.distplot(df.team_batting_h, kde=False)
ax.set_title("Histogram of Team Batting Hits")
plt.show()

#Get outliers
print(get_outliers(df.team_batting_h))

#Plots of batting 1B
ax = sns.boxplot([df.team_batting_1b], orient='v')
ax.set_title("Boxplot of Team Batting 1Bs")
ax.set_ylabel("1Bs")
plt.show()

ax = sns.distplot(df.team_batting_1b, kde=False)
ax.set_title("Histogram of Team Batting 1Bs")
plt.show()

#Get outliers
print(get_outliers(df.team_batting_1b))

#Plots of batting 2B
ax = sns.boxplot([df.team_batting_2b], orient='v')
ax.set_title("Boxplot of Team Batting 2Bs")
ax.set_ylabel("2Bs")
plt.show()

ax = sns.distplot(df.team_batting_2b, kde=False)
ax.set_title("Histogram of Team Batting 2Bs")
plt.show()

#Get outliers
print(get_outliers(df.team_batting_2b))

#Plots of batting 3B
ax = sns.boxplot([df.team_batting_3b], orient='v')
ax.set_title("Boxplot of Team Batting 3Bs")
ax.set_ylabel("3Bs")
plt.show()

ax = sns.distplot(df.team_batting_3b, kde=False)
ax.set_title("Histogram of Team Batting 3Bs")
plt.show()

#Get outliers
print(get_outliers(df.team_batting_3b))

#Plots of batting HR
ax = sns.boxplot([df.team_batting_hr], orient='v')
ax.set_title("Boxplot of Team Batting HRs")
ax.set_ylabel("HRs")
plt.show()

ax = sns.distplot(df.team_batting_hr, kde=False)
ax.set_title("Histogram of Team Batting HRs")
plt.show()

#Get outliers
print(get_outliers(df.team_batting_hr))

#Plots of batting BB
ax = sns.boxplot([df.team_batting_bb], orient='v')
ax.set_title("Boxplot of Team Batting BBs")
ax.set_ylabel("BBs")
plt.show()

ax = sns.distplot(df.team_batting_bb, kde=False)
ax.set_title("Histogram of Team Batting BBs")
plt.show()

#Get outliers
print(get_outliers(df.team_batting_bb))

#Plots of batting HPB
ax = sns.boxplot([df.imp_team_batting_hbp], orient='v')
ax.set_title("Boxplot of Team Batting HBPs")
ax.set_ylabel("HBPs")
plt.show()

ax = sns.distplot(df.imp_team_batting_hbp, kde=False)
ax.set_title("Histogram of Team Batting HBPs")
plt.show()

#Get outliers
print(get_outliers(df.imp_team_batting_hbp))

#Plots of batting SOs
ax = sns.boxplot([df.imp_team_batting_so], orient='v')
ax.set_title("Boxplot of Team Batting SOs")
ax.set_ylabel("SOs")
plt.show()

ax = sns.distplot(df.imp_team_batting_so, kde=False)
ax.set_title("Histogram of Team Batting SOs")
plt.show()

#Get outliers
print(get_outliers(df.imp_team_batting_so))

#Plots of batting OBC
ax = sns.boxplot([df.team_batting_obc], orient='v')
ax.set_title("Boxplot of Team Batting OBC")
ax.set_ylabel("OBC")
plt.show()

ax = sns.distplot(df.team_batting_obc, kde=False)
ax.set_title("Histogram of Team Batting OBC")
plt.show()

#Get outliers
print(get_outliers(df.team_batting_obc))

#Plots of batting Total Bases
ax = sns.boxplot([df.team_batting_totalbases], orient='v')
ax.set_title("Boxplot of Team Batting Total Bases")
ax.set_ylabel("Total Bases")
plt.show()

ax = sns.distplot(df.team_batting_totalbases, kde=False)
ax.set_title("Histogram of Team Batting Total Bases")
plt.show()

#Get outliers
print(get_outliers(df.team_batting_totalbases))

#################################################
#Baserunning
#Plots of baserunning steals
ax = sns.boxplot([df.imp_team_baserun_sb], orient='v')
ax.set_title("Boxplot of Team Basrunning SBs")
ax.set_ylabel("SBs")
plt.show()

ax = sns.distplot(df.imp_team_baserun_sb, kde=False)
ax.set_title("Histogram of Team Basrunning SBs")
plt.show()

#Get outliers
print(get_outliers(df.imp_team_baserun_sb))

#Plots of baserunning caught stealing
ax = sns.boxplot([df.imp_team_baserun_cs], orient='v')
ax.set_title("Boxplot of Team Basrunning CSs")
ax.set_ylabel("CSs")
plt.show()

ax = sns.distplot(df.imp_team_baserun_cs, kde=False)
ax.set_title("Histogram of Team Basrunning CSs")
plt.show()

#Get outliers
print(get_outliers(df.imp_team_baserun_cs))

#################################################
#Fielding
#Plots of fielding errors
ax = sns.boxplot([df.team_fielding_e], orient='v')
ax.set_title("Boxplot of Team Fielding Es")
ax.set_ylabel("Errors")
plt.show()

ax = sns.distplot(df.team_fielding_e, kde=False)
ax.set_title("Histogram of Team Fielding Es")
plt.show()

#Get outliers
print(get_outliers(df.team_fielding_e))

#Plots of fielding double plays
ax = sns.boxplot([df.imp_team_fielding_dp], orient='v')
ax.set_title("Boxplot of Team Fielding DPs")
ax.set_ylabel("DPs")
plt.show()

ax = sns.distplot(df.imp_team_fielding_dp, kde=False)
ax.set_title("Histogram of Team Fielding DPs")
plt.show()

#Get outliers
print(get_outliers(df.imp_team_fielding_dp))

#################################################
#Pitching
#Plots of pitching walks
ax = sns.boxplot([df.team_pitching_bb], orient='v')
ax.set_title("Boxplot of Team Pitching BBs")
ax.set_ylabel("BBs")
plt.show()

ax = sns.distplot(df.team_pitching_bb, kde=False)
ax.set_title("Histogram of Team Pitching BBs")
plt.show()

#Get outliers
print(get_outliers(df.team_pitching_bb))

#Plots of pitching hits
ax = sns.boxplot([df.team_pitching_h], orient='v')
ax.set_title("Boxplot of Team Pitching Hits")
ax.set_ylabel("Hits")
plt.show()

ax = sns.distplot(df.team_pitching_h, kde=False)
ax.set_title("Histogram of Team Pitching Hits")
plt.show()

#Get outliers
print(get_outliers(df.team_pitching_h))

min(get_outliers(df.team_pitching_h))

#Plots of pitching homeruns
ax = sns.boxplot([df.team_pitching_hr], orient='v')
ax.set_title("Boxplot of Team Pitching HRs")
ax.set_ylabel("HRs")
plt.show()

ax = sns.distplot(df.team_pitching_hr, kde=False)
ax.set_title("Histogram of Team Pitching HRs")
plt.show()

#Get outliers
print(get_outliers(df.team_pitching_hr))

#Plots of pitching strikeouts
ax = sns.boxplot([df.imp_team_pitching_so], orient='v')
ax.set_title("Boxplot of Team Pitching SOs")
ax.set_ylabel("SOs")
plt.show()

ax = sns.distplot(df.imp_team_pitching_so, kde=False)
ax.set_title("Histogram of Team Pitching SOs")
plt.show()

#Get outliers
print(get_outliers(df.imp_team_pitching_so))

#Plots of pitching OBC
ax = sns.boxplot([df.team_pitching_obc], orient='v')
ax.set_title("Boxplot of Team Pitching OBC")
ax.set_ylabel("OBC")
plt.show()

ax = sns.distplot(df.team_pitching_obc, kde=False)
ax.set_title("Histogram of Team Pitching OBC")
plt.show()

#Get outliers
print(get_outliers(df.team_pitching_obc))

##########################################################
def fixOutliers(onePdSer, otherPdSer, q1, q2):
    newpdSer = onePdSer.copy()
    for val in range(0, len(onePdSer)):
        if onePdSer[val] > otherPdSer.quantile(q1):
            newpdSer[val] = otherPdSer.quantile(q1)
        
        if onePdSer[val] < otherPdSer.quantile(q2):
            newpdSer[val] = otherPdSer.quantile(q2)
    
    return newpdSer

df.target_wins = fixOutliers(df.target_wins, df.target_wins, .99, .01)
df.team_batting_h = fixOutliers(df.team_batting_h, df.team_batting_h, .99, .01)
df.team_batting_1b = fixOutliers(df.team_batting_1b, df.team_batting_1b, .99, .01)
df.team_batting_2b = fixOutliers(df.team_batting_2b, df.team_batting_2b, .99, .01)
df.team_batting_3b = fixOutliers(df.team_batting_3b, df.team_batting_3b, .99, .01)
df.team_batting_bb = fixOutliers(df.team_batting_bb, df.team_batting_bb, .99, .01)
df.imp_team_batting_so = fixOutliers(df.imp_team_batting_so, df.imp_team_batting_so, .99, .01)
df.imp_team_baserun_sb = fixOutliers(df.imp_team_baserun_sb, df.imp_team_baserun_sb, .99, .01)
df.imp_team_baserun_cs = fixOutliers(df.imp_team_baserun_cs, df.imp_team_baserun_cs, .99, .01)
df.team_fielding_e = fixOutliers(df.team_fielding_e, df.team_fielding_e, .95, .05)
df.imp_team_fielding_e = fixOutliers(df.imp_team_fielding_dp, df.imp_team_fielding_dp, .99, .01)
df.team_pitching_bb = fixOutliers(df.team_pitching_bb, df.team_pitching_bb, .99, .01)
df.team_pitching_h = fixOutliers(df.team_pitching_h, df.team_pitching_h, .95, .05)
df.imp_team_pitching_so = fixOutliers(df.imp_team_pitching_so, df.imp_team_pitching_so, .99, .01)

df.team_batting_obc = df.team_batting_bb + df.team_batting_h + df.imp_team_batting_hbp
df.team_batting_totalbases = 1*df.team_batting_1b + 2*df.team_batting_2b + 3*df.team_batting_3b + 4*df.team_batting_hr
df.team_pitching_obc = df.team_pitching_bb + df.team_pitching_h

#Collinearity Check
corrs = df.corr()
print(corrs)

#Get heatmap
ax = plt.axes()
sns.heatmap(corrs, 
            xticklabels=corrs.columns.values,
            yticklabels=corrs.columns.values, ax = ax)

ax.set_title('Moneyball Correlation Heatmap')
plt.show()

#sns.pairplot(df)  

###########################################################
#Regression 1
#gather features
features = "+".join(df.columns[1:32])
y = df.target_wins

result1 = ols('target_wins ~' + features, data=df).fit()
print(result1.summary())

#anova table
aov_table_1 = sm.stats.anova_lm(result1, typ=2)
print(aov_table_1)

#Get predictions and residuals
x1_predictions = result1.predict()
x1_residuals = y - x1_predictions

#Plots
#Residuals vs Fitted
fig = plt.figure()
plt.scatter(x=x1_predictions, y= x1_residuals)
fig.suptitle('Multiple Model Residuals vs Fitted Wins Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

#QQPlot
stats.probplot(x1_residuals, dist="norm", plot=plt)
plt.show()

print(mean_absolute_error(y, x1_predictions)) #9.36075260602

#Gets the columns without collinearity
def getBestCols(inpDF, inpYString, thresh = 5):
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
        firstModel = ols(inpYString + " ~ " + testFeatures1, data=testDF1).fit()
        
        #Drop other and create model
        testDF2 = inpDF.drop(colToTest2, 1)
        tempList2 = list(testDF2.columns.values)
        tempList2.remove(inpYString)
        testFeatures2 = "+".join(tempList2)
        secondModel = ols(inpYString + " ~ " + testFeatures2, data=testDF2).fit()
        
        #Pick better model and recurse
        if firstModel.rsquared_adj > secondModel.rsquared_adj:
            return getBestCols(testDF1, inpYString)
        elif secondModel.rsquared_adj > firstModel.rsquared_adj:
            return getBestCols(testDF2, inpYString)
        else:
            return getBestCols(testDF1, inpYString)

##################################
theBestCols = getBestCols(df, "target_wins").values
index = np.argwhere(theBestCols=="Intercept")
theBestCols = np.delete(theBestCols, index)
theBestCols = np.append(theBestCols, "target_wins")
df_sub = df[theBestCols]

#Double check the results of the above function
tempList = list(df_sub.columns.values)
tempList.remove("target_wins")
testFeatures = "+".join(tempList)
y, X = dmatrices('target_wins ~' + testFeatures, df_sub, return_type='dataframe')
# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Features"] = X.columns

print(vif)

#Looks good!

#Mulitple Regression 2

#gather features
y = df_sub.target_wins

result2 = ols('target_wins ~' + testFeatures, data=df_sub).fit()
print(result2.summary())

#anova table
aov_table_2 = sm.stats.anova_lm(result2, typ=2)
print(aov_table_2)

#Get predictions and residuals
x2_predictions = result2.predict()
x2_residuals = y - x2_predictions

#Plots
#Residuals vs Fitted
fig = plt.figure()
plt.scatter(x=x2_predictions, y= x2_residuals)
fig.suptitle('Multiple Subset Model Residuals vs Fitted Wins Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

#QQPlot
stats.probplot(x2_residuals, dist="norm", plot=plt)
plt.show()

print(mean_absolute_error(y, x2_predictions)) #10.0233086535


###############################################################
#PCA
df_sub2 = df.copy(deep=False)
del(df_sub2["target_wins"])
# work with standard scores for all pca variables
# standard scores have zero mean and unit standard deviation
pca_data = preprocessing.scale(df_sub2.as_matrix())
pca = PCA()
pca.fit(pca_data)

# show summary of pca solution
pca_explained_variance = pca.explained_variance_ratio_
print('Proportion of variance explained:', pca_explained_variance)

# note that principal components analysis corresponds
# to finding eigenvalues and eigenvectors of the correlation matrix
pca_data_cormat = np.corrcoef(pca_data.T)

#get eigen values and vectors
eig_vals, eig_vecs = np.linalg.eigh(pca_data_cormat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#Check unit length 1
#for ev in eig_vecs:
    #np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

print('Linear algebra demonstration: Proportion of variance explained: ',
    eig_vals/eig_vals.sum())

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

sorted_eig_vals = []

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    sorted_eig_vals.append(i[0])
    print(i[0])

#Scree Plot
fig = plt.figure()
plt.scatter(range(1,26), sorted_eig_vals)
plt.plot(range(1,26), sorted_eig_vals)
fig.suptitle('Scree Plot of Different Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalue')
plt.show()

#Get cumulative variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

#Explained Variance Plot
fig = plt.figure()
plt.scatter(range(1,26), cum_var_exp)
plt.plot(range(1,26), cum_var_exp)
fig.suptitle('Explained Variance by Different Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance in Pecent')
plt.show()

# show the scree plot for the pricipal component analysis
plt.bar(np.arange(len(pca_explained_variance)), pca_explained_variance, 
    color = 'blue', alpha = 0.5, align = 'center')
plt.title('PCA Proportion of Total Variance')   
plt.xlabel('Principal Components')
plt.ylabel('Proportion of Total Variance')
plt.show()

# compute full set of principal components (scores)
C = pca.transform(pca_data)

# add first seven principal component scores to the original data frame
df_sub2['pca1'] = C[:,0]
df_sub2['pca2'] = C[:,1]
df_sub2['pca3'] = C[:,2]
df_sub2['pca4'] = C[:,3]
df_sub2['pca5'] = C[:,4]
df_sub2['pca6'] = C[:,5]
df_sub2['pca7'] = C[:,6]

df_sub2["target_wins"] = df.target_wins

#Mulitple Regression 3

#gather features
features3 = "+".join(df_sub2.columns[25:32])
y = df_sub2.target_wins

result3 = ols('target_wins ~' + features3, data=df_sub2).fit()
print(result3.summary())

#anova table
aov_table_3 = sm.stats.anova_lm(result3, typ=2)
print(aov_table_3)

#Get predictions and residuals
x3_predictions = result3.predict()
x3_residuals = y - x3_predictions

#Plots
#Residuals vs Fitted
fig = plt.figure()
plt.scatter(x=x3_predictions, y= x3_residuals)
fig.suptitle('Multiple PCA Model Residuals vs Fitted Wins Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

#QQPlot
stats.probplot(x3_residuals, dist="norm", plot=plt)
plt.show()

print(mean_absolute_error(y, x3_predictions)) #10.6285652869

#######################################################################
#Regression 4

#gather features
df_sub3 = df.copy()
y = df_sub3.target_wins

df_sub3["log_team_batting_totalbases"] = np.log(df_sub3.team_batting_totalbases)
df_sub3["log_team_batting_obc"] = np.log(df_sub3.team_batting_obc)
df_sub3["log_team_batting_h"] = np.log(df_sub3.team_batting_h)
df_sub3["log_team_batting_bb"] = np.sqrt(df_sub3.team_batting_bb)
#df_sub3["sqrt_imp_team_batting_so"] = np.log(df_sub3.imp_team_batting_so)
#df_sub3["log_imp_team_pitching_so"] = np.log(df_sub3.imp_team_pitching_so)
#df_sub3["log_team_pitching_obc"] = np.log(df_sub3.team_pitching_obc)
df_sub3["sqrt_team_fielding_e"] = np.sqrt(df_sub3.team_fielding_e)
df_sub3["sqrt_imp_team_fielding_dp"] = np.sqrt(df_sub3.imp_team_fielding_dp)

df_sub3["log_imp_team_baserun_cs"] = np.log(df_sub3.imp_team_baserun_cs)

features4 = 'team_batting_bb + team_batting_2b + team_batting_hr + log_team_batting_h + log_team_batting_totalbases + log_team_batting_obc + sqrt_team_fielding_e + log_imp_team_baserun_cs + m_team_baserun_cs + imp_team_baserun_sb + m_team_baserun_sb + m_team_batting_hbp + imp_team_batting_so + m_team_batting_so + sqrt_imp_team_fielding_dp + m_team_fielding_dp + m_team_pitching_so'  

result5 = ols('target_wins ~' + features4, data=df_sub3).fit()
print(result5.summary())

#anova table
aov_table_5 = sm.stats.anova_lm(result5, typ=2)
print(aov_table_5)

#Get predictions and residuals
x5_predictions = result5.predict()
x5_residuals = y - x5_predictions

#Plots
#Residuals vs Fitted
fig = plt.figure()
plt.scatter(x=x5_predictions, y= x5_residuals)
fig.suptitle('Multiple Model Residuals vs Fitted Wins Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

#QQPlot
stats.probplot(x5_residuals, dist="norm", plot=plt)
plt.show()

print(mean_absolute_error(y, x5_predictions)) #9.2021204627

#####################################################################################
#Regression 5
fakeOPS = df_sub3.team_batting_totalbases + df_sub3.team_batting_obc
#q25 = 3888.0
#q50 = 4157.0
#q75 = 4399.0

df_sub3["team_batting_fakeops"] = df_sub3.team_batting_totalbases + df_sub3.team_batting_obc

#df_sub3["fakeops_bin1"] = 0
#
#for indx in range(0, len(fakeOPS)):
#    binn = 0
#    val = fakeOPS[indx]
#    if val < q25:
#        binn = 0
#    elif q25 <= val < q50:
#        binn = 0
#    elif q50 <= val < q75:
#        binn = 1
#    elif q75 <= val:
#        binn = 1
#    df_sub3.fakeops_bin1[indx] = binn

#df_sub3["fakeops_bin2"] = 0
#for indx in range(0, len(fakeOPS)):
#    binn = 0
#    val = (fakeOPS[indx] - min(fakeOPS))/(max(fakeOPS) - min(fakeOPS))
#    binn = int(math.floor(val*4))
#    df_sub3.fakeops_bin2[indx] = binn
    
features5 = 'team_batting_fakeops + team_batting_bb + team_batting_2b + team_batting_hr + log_team_batting_h + log_team_batting_totalbases + log_team_batting_obc + sqrt_team_fielding_e + log_imp_team_baserun_cs + m_team_baserun_cs + imp_team_baserun_sb + m_team_baserun_sb + m_team_batting_hbp + imp_team_batting_so + m_team_batting_so + sqrt_imp_team_fielding_dp + m_team_fielding_dp + m_team_pitching_so'  

result5 = ols('target_wins ~' + features5, data=df_sub3).fit()
print(result5.summary())

#anova table
aov_table_5 = sm.stats.anova_lm(result5, typ=2)
print(aov_table_5)

#Get predictions and residuals
x5_predictions = result5.predict()
x5_residuals = y - x5_predictions

#Plots
#Residuals vs Fitted
fig = plt.figure()
plt.scatter(x=x5_predictions, y= x5_residuals)
fig.suptitle('Multiple Model Residuals vs Fitted Wins Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

#QQPlot
stats.probplot(x5_residuals, dist="norm", plot=plt)
plt.show()

print(mean_absolute_error(y, x5_predictions)) #9.19323977893


############################################################################
test = pd.read_csv(r'moneyball_test.csv')
test1 = test.copy(deep=False)

for indx in cols:
    if naCols[indx] == True:
        newName = "IMP_" + indx
        newInd = "m_" + indx
        m = np.median(train[indx][train[indx] > 0])
        test1[newName] = test[indx].fillna(m)
        test1[newInd] = test[indx].isnull().astype(int)

test1["TEAM_BATTING_1B"] = test1.TEAM_BATTING_H - test1.TEAM_BATTING_2B - test1.TEAM_BATTING_3B - test1.TEAM_BATTING_HR
test1.columns = [s.lower() for s in test1.columns]

test1.team_batting_h = fixOutliers(test1.team_batting_h, train1.team_batting_h, .99, .01)
test1.team_batting_1b = fixOutliers(test1.team_batting_1b, train1.team_batting_1b, .99, .01)
test1.team_batting_2b = fixOutliers(test1.team_batting_2b, train1.team_batting_2b, .99, .01)
test1.team_batting_3b = fixOutliers(test1.team_batting_3b, train1.team_batting_3b, .99, .01)
test1.team_batting_bb = fixOutliers(test1.team_batting_bb, train1.team_batting_bb, .99, .01)
test1.imp_team_batting_so = fixOutliers(test1.imp_team_batting_so, train1.imp_team_batting_so, .99, .01)
test1.imp_team_baserun_sb = fixOutliers(test1.imp_team_baserun_sb, train1.imp_team_baserun_sb, .99, .01)
test1.imp_team_baserun_cs = fixOutliers(test1.imp_team_baserun_cs, train1.imp_team_baserun_cs, .99, .01)
test1.team_fielding_e = fixOutliers(test1.team_fielding_e, train1.team_fielding_e, .95, .05)
test1.imp_team_fielding_e = fixOutliers(test1.imp_team_fielding_dp, train1.imp_team_fielding_dp, .99, .01)
test1.team_pitching_bb = fixOutliers(test1.team_pitching_bb, train1.team_pitching_bb, .99, .01)
test1.team_pitching_h = fixOutliers(test1.team_pitching_h, train1.team_pitching_h, .95, .05)
test1.imp_team_pitching_so = fixOutliers(test1.imp_team_pitching_so, train1.imp_team_pitching_so, .99, .01)

test1.team_batting_obc = test1.team_batting_bb + test1.team_batting_h + test1.imp_team_batting_hbp
test1.team_batting_totalbases = 1*test1.team_batting_1b + 2*test1.team_batting_2b + 3*test1.team_batting_3b + 4*test1.team_batting_hr
test1.team_pitching_obc = test1.team_pitching_bb + test1.team_pitching_h

test1["log_team_batting_totalbases"] = np.log(test1.team_batting_totalbases)
test1["log_team_batting_obc"] = np.log(test1.team_batting_obc)
test1["log_team_batting_h"] = np.log(test1.team_batting_h)
test1["log_team_batting_bb"] = np.sqrt(test1.team_batting_bb)
test1["sqrt_team_fielding_e"] = np.sqrt(test1.team_fielding_e)
test1["sqrt_imp_team_fielding_dp"] = np.sqrt(test1.imp_team_fielding_dp)
test1["log_imp_team_baserun_cs"] = np.log(test1.imp_team_baserun_cs)
test1["team_batting_fakeops"] = test1.team_batting_totalbases + test1.team_batting_obc

predictions1 = result5.predict(test1)
print(predictions1)

d = {'p_target_wins': predictions1}
df1 = test1[['index']]
df2=pd.DataFrame(data=d)
your_file = pd.concat([df1,df2],axis = 1, join_axes=[df1.index])

#Submit your file as csv using the following code to save on your computer
your_file.to_csv('booth_alexander_unit01_moneyball_test_predictions.csv', index=False) 