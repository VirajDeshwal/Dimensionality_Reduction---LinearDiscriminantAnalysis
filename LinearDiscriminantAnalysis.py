#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:20:51 2018

@author: virajdeshwal
"""
'''Linear Discriminant Analysis is a Supervised Learning '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

file = pd.read_csv('Wine.csv')
#we are including the two index from our dataset and finding the corelation between them.

X = file.iloc[:,0:13].values
y= file.iloc[:,13].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)


#we need to do the feature scaling in the logistic regression to get the accurate prediction.

from sklearn.preprocessing import StandardScaler

scaling = StandardScaler()

x_train = scaling.fit_transform(x_train)
x_test = scaling.transform(x_test)

#Applying LDA to reduce the dimensions of the independent variable.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
'''we have to define the no. LDA for most variance. 
And check which components explain the most variance in the given dataset.
We want 2 independent variables but for now to check which are best. We will enter None.
And later replace it with the no. of top components'''


lda = LDA(n_components=2)
'''as LDA is a superviese learning model. We have to fit the dependent variable(y) as well.'''
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


from sklearn.metrics import confusion_matrix


conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
 
plt.imshow(conf_matrix)
plt.title('Graphical representation of Prediction of how many people will buy the SUV')
plt.xlabel('AGE')
plt.ylabel('Estimated Salary')
plt.show()




# Visualising the Training set results
'''As we have 3 different classes of Customer_segment. We will use 3 different colors'''
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('LDA (Training set)')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.legend()
plt.show()

# Visualising the Test set results

from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('LDA(Test set)')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy = accuracy*100
print('\n\n\n Hence the accuracy of the Linear Regression after LDA is:',accuracy)
print('\n\n Done :)')