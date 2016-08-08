# -*- coding: utf-8 -*-
"""
Created on Tue May 10 23:28:33 2016

@author: PikeZhao
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 08 21:57:36 2016

@author: PikeZhao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn as sk
from sklearn import datasets, linear_model

# Decision Tree from Spark MLlib
Y_test_actual = np.array([80.59	,70.52,69.45,73.12,73.42,79.28,74.09,80.24,65.65,80.15,59.58,51.30,76.92,74.83,69.55])
Y_test_preds = np.array([81.53,73.16,70.81,73.16,73.16,73.16,73.16,79.80,73.16,81.53,58.80,61.61,73.16,66.92,81.53])
plt.scatter(Y_test_actual,Y_test_preds)
plt.grid(True)
plt.xlabel("Train:life expectancy from data")
plt.ylabel("Train:life expectancy predicted")
plt.plot([0,100],[0,100])
plt.ylim([40,85])
plt.xlim([40,85])
plt.title("Tested life expectancy vs predicted for Spark MLlib Decision Tree")

# Load the diabetes dataset
df = pandas.read_csv('D://Study//Other_MS//HarvardExtensionSchool//E63-BigDataAnalytics//FinalProject//MyProject//data//000000_0_r5.csv')
y = np.array(df['Life1'].tolist())
X = df.drop('Life1',axis = 1)
X_train,X_test,Y_train,Y_test = sk.cross_validation.train_test_split(X,y,test_size = 0.2)
regr = linear_model.LinearRegression()
clf = regr.fit(X_train, Y_train)

pandas.DataFrame(zip(X_train.columns,clf.coef_),columns=['features','estimatedCoefficients'])

# Trained life expectancy vs predicted
plt.scatter(Y_train,clf.predict(X_train))
plt.grid(True)
plt.xlabel("Train:life expectancy from data")
plt.ylabel("Train:life expectancy predicted")
plt.plot([0,100],[0,100])
plt.ylim([40,85])
plt.xlim([40,85])
plt.title("Trained life expectancy vs predicted")

# Tested life expectancy vs predicted
plt.scatter(Y_test,clf.predict(X_test))
plt.xlabel("Test:life expectancy from data")
plt.ylabel("Test:life expectancy predicted")
plt.grid(True)
plt.ylim([40,85])
plt.xlim([40,85])
# plt.hlines(y=0, xmin=0, xmax= 100)
plt.plot([0,100],[0,100])
plt.title("Tested life expectancy vs predicted")

# Residual plt
plt.scatter(clf.predict(X_train), clf.predict(X_train) - Y_train, c='b',s=40, alpha = 0.5)
plt.scatter(clf.predict(X_test),clf.predict(X_test) - Y_test, c='g',s=40)
plt.hlines(y=0, xmin=0, xmax= 100)
plt.title('Residual plt using training (blue) and test (green) data')
plt.ylabel("Residuals")
plt.grid(True)


#--------------------Decision Tree
from sklearn import tree
clf_dt = tree.DecisionTreeClassifier()
clf_dt = clf_dt.fit(X_train, Y_train)

plt.scatter(Y_train,clf_dt.predict(X_train))
plt.grid(True)
plt.xlabel("Train_dt:life expectancy from data")
plt.ylabel("Train_dt:life expectancy predicted")
plt.plot([0,100],[0,100])
plt.ylim([40,85])
plt.xlim([40,85])
plt.title("Trained_dt life expectancy vs predicted")

plt.scatter(Y_test,clf_dt.predict(X_test))
plt.xlabel("Test:life expectancy from data")
plt.ylabel("Test:life expectancy predicted")
plt.grid(True)
plt.ylim([40,85])
plt.xlim([40,85])
# plt.hlines(y=0, xmin=0, xmax= 100)
plt.plot([0,100],[0,100])
plt.title("Tested_dt life expectancy vs predicted")