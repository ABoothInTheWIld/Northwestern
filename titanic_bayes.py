# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

df = pd.read_csv("C:/Users/Alexander/Documents/Northwestern/Winter 2017/MSPA 400/Python/Session 4/train.csv")

sub_df = df[['Sex', 'Survived']]

#P(A)
p_survived = len(sub_df[sub_df['Survived'] == 1])/float(len(sub_df))
p_died = len(sub_df[sub_df['Survived'] == 0])/float(len(sub_df))
#p_died = 1 - p_survived

#P(B)
p_male = len(sub_df[sub_df['Sex'] == 'male'])/float(len(sub_df))
p_female = len(sub_df[sub_df['Sex'] == 'female'])/float(len(sub_df))
#p_female = 1 - p_male

#P(B|A)
# = P(A|B)*P(B)/P(A)
#Definition of Joint Probability:
# = P(A & B)/P(A)
p_male_given_survived = len(sub_df.ix[(sub_df['Survived'] == 1) & (sub_df['Sex'] == 'male')])/float(len(sub_df[sub_df['Survived'] == 1]))
p_female_given_survived = len(sub_df.ix[(sub_df['Survived'] == 1) & (sub_df['Sex'] == 'female')])/float(len(sub_df[sub_df['Survived'] == 1]))
#p_female_given_survived = 1 - p_male_given_survived

#Bayes Theorem
#P(A|B) = P(B|A)*P(A)/P(B)
p_survived_given_male = (p_male_given_survived * p_survived)/p_male
p_survived_given_female = (p_female_given_survived * p_survived)/p_female

#Note:
#p_survived_given_female != 1 - p_survived_given_male
#Because of different denominators
#1 - p_survived_given_male = p_died_given_male
#p_died_given_male = (p_male_given_died * p_died)/p_male
#This is the same for females

#check
#P(A|B)
#Definition of Joint Probability:
# = P(B & A)/P(B)
p_survived_given_male_check = len(sub_df.ix[(sub_df['Sex'] == 'male') & (sub_df['Survived'] == 1)])/float(len(sub_df[sub_df['Sex'] == 'male']))
p_survived_given_female_check = len(sub_df.ix[(sub_df['Sex'] == 'female') & (sub_df['Survived'] == 1)])/float(len(sub_df[sub_df['Sex'] == 'female']))

print("Given you were a male, you had about an %.2f percent chance of surviving the Titanic." % (p_survived_given_male * 100))
print("Given you were a female, you had about an %.2f percent chance of surviving the Titanic." % (p_survived_given_female * 100))

#Given you were a male, you had about an 18.89 percent chance of surviving the Titanic.
#Given you were a female, you had about an 74.20 percent chance of surviving the Titanic.