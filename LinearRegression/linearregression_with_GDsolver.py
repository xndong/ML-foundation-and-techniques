# -*- coding: utf-8 -*-
"""

@author: DongXiaoning
"""

import numpy as np
import random
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def getData(dataSet):
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:, :-1] = dataSet[:, :-1]
    trainLabel = dataSet[:, -1]
    return trainData, trainLabel


def GradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrans = x.transpose()
    for i in range(maxIterations):
        hypothesis = np.dot(x, theta)
        error = hypothesis - y
        gradient = np.dot(xTrans, error) / m
        theta = theta - alpha * gradient
        #print(theta)
        print(gradient)
    return theta


def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n + 1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)
    return yP


dataPath = r"C:\Users\DongXiaoning\Downloads\Algorithms in Machine Learning Foundation and Techniques\LinearRegression\house.csv"
# dataPath = r"\\192.168.1.18\Users\DongXiaoning\Downloads\Algorithms in Machine Learning Foundation and Techniques\LinearRegression\house.csv"
dataSet = genfromtxt(dataPath, delimiter=',')
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)
alpha = 0.01
maxIteration = 5000
theta = GradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print('\n')
print(predict(x, theta))
print('\n')

#visualization
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('x0 axis')
ax.set_ylabel('x1 axis')
ax.set_zlabel('y axis')
ax.scatter(trainData[:, 0], trainData[:, 1], trainLabel, c='r')

x0 = np.linspace(-2, 5, 20)
x1 = np.linspace(0, 7, 20)
X0, X1 = np.meshgrid(x0, x1)
y = X0 * theta[0] + X1 * theta[1] + theta[2]
ax.plot_wireframe(X0, X1, y, cstride=1, rstride=1, color='b')
plt.show()
