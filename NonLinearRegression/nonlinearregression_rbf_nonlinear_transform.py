# -*- coding: utf-8 -*-
"""

@author: DongXiaoning
"""

# In essence, this is a special case of RBF network, where both center's parameters and center numbers are given.
# While in RBF network, center's parameters are learned and center numbers are given.
# Also, RBF(Guassian) Kernal SVM is smilar to RBF network, where center's parameter are given and center numbers are given by SVs(learned).
# General and high ability: rbf network = RBF Kernal SVM > nonlinear rbf transformation(this python program)

import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rbf_nonlinearTransform(rawtrainData):
    row,col = np.shape(rawtrainData)
    center_numbers = 15
    sigma = 1
    np_array = np.empty((0,center_numbers), float)
    for i in range(row):
        arr = compute_RBF(rawtrainData[i,:], center_numbers, sigma)
        arr = arr.reshape(1,center_numbers)
        np_array = np.concatenate((np_array, arr), axis=0)
    return np_array

def compute_RBF(x, center_numbers, sigma):
    dim = np.shape(x)[0]
    centers = np.random.rand(center_numbers, dim)  # 随机选取center_numbers个center
    sigmas = np.array([sigma + (i % 4) * 0.5 for i in range(center_numbers)])
    alist = []
    for i in range(center_numbers):
        trnorms1 = np.dot(x,x)
        trnorms2 = np.dot(centers[i],centers[i])
        trnorms3 = np.dot(x,centers[i])
        two_norm = trnorms1 + trnorms2 - 2 * trnorms3
        coefficient =  - 1./(2 * np.power(sigmas[i], 2))
        from_center_i = np.exp(coefficient * two_norm)
        alist.append(from_center_i)
    return np.array(alist)

def GradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrans = x.transpose()
    for i in range(maxIterations):
        hypothesis = np.dot(x, theta)
        error = hypothesis - y
        gradient = np.dot(xTrans, error) / m
        theta = theta - alpha * gradient
        #print(theta)
        print(gradient)
    return theta,gradient

def stochasticGradientDescent(x, y, theta, alpha, m, maxIterations):
    for i in range(maxIterations):
        index = i % m
        hypothesis = np.dot(x[index,:], theta)
        error = hypothesis - y[index]
        stochastic_gradient = error * x[index, :]
        theta = theta - alpha * stochastic_gradient
        #print(theta)
        print(stochastic_gradient)
    return theta,stochastic_gradient


def predict(x, theta):
    yP = np.dot(x, theta)
    return yP

if __name__ == '__main__':

    # generate dataset from scikit-learn instead of using local files.
    # It is more convenient because it avoid reading local file to our program
    boston_dataset = load_boston()
    
    diabetes = load_diabetes()
    print(diabetes.keys())
    print(diabetes['feature_names'])
    # To be simple, we only use first 6 features, index 0-5
    rawtrainData = diabetes['data'][:-100, :6]
    rawtrainLabel = diabetes['target'][:-100]
    rawtestData = diabetes['data'][-100:, :6]
    rawtestLabel = diabetes['target'][-100:]
    
    trainData = rbf_nonlinearTransform(rawtrainData)
    trainLabel = rawtrainLabel
    testData = rbf_nonlinearTransform(rawtestData)
    testLabel = rawtestLabel
    
    m, n = np.shape(trainData)
    theta = np.ones(n)
    alpha = 0.1
    maxIteration = 50000
    theta,gradient = GradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
    # theta = [34.81356448,-97.69060875,794.05206073,382.50087972,174.98727396,-159.82892325,555.04962131,332.48352938,585.48068579,655.77553744,
    # 659.71413794,147.2941835]
    # theta = np.asarray(theta)
    print('\n')
    print(f'Final theta value is {theta}')
    print(f'Final gradient value is {gradient}')
    print('\n')
    predictLabel_trainData = predict(trainData, theta)
    predictLabel_testData = predict(testData, theta)
    # print(predictLabel_trainData)
    # print(predictLabel_testData)
    print('\n')
    
    # visualization
    X = np.arange(100)
    plt.plot(X,testLabel,color='b')
    plt.plot(X,predictLabel_testData,color='g')
    plt.show()

    X = np.arange(342)
    plt.plot(X,trainLabel,color='b')
    plt.plot(X + 0.5,predictLabel_trainData,color='g')
    plt.show()