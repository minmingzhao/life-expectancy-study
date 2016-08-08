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

# Load the diabetes dataset
df = pandas.read_csv('D://Study//Other_MS//HarvardExtensionSchool//E63-BigDataAnalytics//FinalProject//MyProject//data//000000_0_r5.csv')
y = np.array(df['Life1'].tolist())
X = df.drop('Life1',axis = 1)
X_train,X_test,Y_train,Y_test = sk.cross_validation.train_test_split(X,y,test_size = 0.2)
print X_train.shape
print Y_train.shape
print Y_test.shape
print X_test.shape

regr = linear_model.LinearRegression()
clf = regr.fit(X_train, Y_train)
print "The coefficients are: ", clf.coef_
print "The intercept is ", clf.intercept_
# clf.decision_function(X)
print "Train data: coefficient of determination R^2 of the prediction ", clf.score(X_train,Y_train)
print "Test data: coefficient of determination R^2 of the prediction ", clf.score(X_test,Y_test)

pandas.DataFrame(zip(X_train.columns,clf.coef_),columns=['features','estimatedCoefficients'])
#clf.predict(X_test)[0:5]
#Y_test[0:5]
#fig,ax = plt.subplots()
plt.scatter(Y_train,clf.predict(X_train))
plt.grid(True)
plt.xlabel("Train:life expectancy from data")
plt.ylabel("Train:life expectancy predicted")
plt.plot([0,100],[0,100])
plt.ylim([40,85])
plt.xlim([40,85])
plt.title("Trained life expectancy vs predicted")

plt.scatter(Y_test,clf.predict(X_test))
plt.xlabel("Test:life expectancy from data")
plt.ylabel("Test:life expectancy predicted")
plt.grid(True)
plt.ylim([40,85])
plt.xlim([40,85])
# plt.hlines(y=0, xmin=0, xmax= 100)
plt.plot([0,100],[0,100])
plt.title("Tested life expectancy vs predicted")

print "Fit a model X_train, and calculate MSE with Y_train:", np.mean((Y_train-clf.predict(X_train))** 2)
print "Fit a model X_train, and calculate MSE with X_test, Y_test:", np.mean((Y_test-clf.predict(X_test))**2)

print "Fit a model X_train, and calculate Root Mean Squared Log Error (RMSE) with Y_train:", np.mean((np.log(Y_train + 1) - np.log(clf.predict(X_train) + 1))**2)
print "Fit a model X_train, and calculate Root Mean Squared Log Error (RMSE) with X_test, Y_test:", np.mean((np.log(Y_test + 1) - np.log(clf.predict(X_test) + 1))**2)

print "Fit a model X_train, and calculate Mean Absolute Error (MAE) with Y_train:", np.mean(np.abs(Y_train - clf.predict(X_train)))
print "Fit a model X_train, and calculate Mean Absolute Error (MAE) with X_test, Y_test:", np.mean(np.abs(Y_test - clf.predict(X_test)))

plt.scatter(clf.predict(X_train), clf.predict(X_train) - Y_train, c='b',s=40, alpha = 0.5)
plt.scatter(clf.predict(X_test),clf.predict(X_test) - Y_test, c='g',s=40)
plt.hlines(y=0, xmin=0, xmax= 100)
plt.title('Residual plt using training (blue) and test (green) data')
plt.ylabel("Residuals")
plt.grid(True)

F,pval = sk.feature_selection.f_regression(X_train, Y_train, center=True)
pandas.DataFrame(zip(X_train.columns,F,pval),columns=['features','ANOVA-F','ANOVA-pval'])

cov = sk.covariance.empirical_covariance(X_train)

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


print "Fit a decision tree model X_train, and calculate MSE with Y_train:", np.mean((Y_train-clf_dt.predict(X_train))** 2)
print "Fit a decision tree model X_train, and calculate MSE with X_test, Y_test:", np.mean((Y_test-clf_dt.predict(X_test))**2)

print "Fit a decision tree model X_train, and calculate Root Mean Squared Log Error (RMSE) with Y_train:", np.mean((np.log(Y_train + 1) - np.log(clf_dt.predict(X_train) + 1))**2)
print "Fit a decision tree model X_train, and calculate Root Mean Squared Log Error (RMSE) with X_test, Y_test:", np.mean((np.log(Y_test + 1) - np.log(clf_dt.predict(X_test) + 1))**2)

print "Fit a decision tree model X_train, and calculate Mean Absolute Error (MAE) with Y_train:", np.mean(np.abs(Y_train - clf_dt.predict(X_train)))
print "Fit a decision tree model X_train, and calculate Mean Absolute Error (MAE) with X_test, Y_test:", np.mean(np.abs(Y_test - clf_dt.predict(X_test)))

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydot 
dot_data = StringIO()  

with open("D://Study//Other_MS//HarvardExtensionSchool//E63-BigDataAnalytics//FinalProject//MyProject//iris.dot", 'w') as f:
    f = tree.export_graphviz(clf_dt, out_file=f)
import os
os.unlink('iris.dot')
    
tree.export_graphviz(clf_dt, out_file=dot_data) 
tree.export_graphviz(clf_dt, out_file=dot_data,feature_names=X_train.columns,class_names="Life", filled=True, rounded=True,special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())  