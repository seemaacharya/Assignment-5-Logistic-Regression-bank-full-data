# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:34:43 2021

@author: DELL
"""
#Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
bank = pd.read_csv("bank-full.csv",sep=';')
bank.head()
#list of columns for reference
bank.columns

#Select the columns
columns = ['age', 'balance','duration','campaign', 'y']
bank1 = bank[columns]
bank1.info()

pd.crosstab(bank1.age,bank1.y).plot(kind = 'line')
#Here graph shows the age group b/w 20-60 has more rejection of application while 60-90 almost everybody
sns.boxplot(data= bank1, orient= 'y')

bank1['outcome'] = bank1.y.map({'no':0, "yes":1})
bank1.tail(10)
bank1.boxplot(column='age', by='outcome')
#probably not a great feature since lot of outliers

feature_col= ['age','balance','duration','campaign']
output_target=['outcome']
x = bank1[feature_col]
y= bank1[output_target]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x,y)
lr.coef_
lr.predict_proba (x)
y_pred = lr.predict(x)
y_pred

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print (confusion_matrix)

#Accuracy score
accuracy = (39342+854)/(39342+854+4435+580)
accuracy
#88.90%

import matplotlib.pyplot as plt
sns.heatmap(confusion_matrix,annot = True)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')