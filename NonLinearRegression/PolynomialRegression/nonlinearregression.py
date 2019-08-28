# -*- coding: utf-8 -*-
"""

@author: DongXiaoning
"""
#console: autopep8 --in-place --aggressive --aggressive nonlinearregression.py
#console: yapf -h

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes


def nonlinearTransform(rawtrainData):
    arr1 = rawtrainData[:, :]
    arr2 = np.multiply(rawtrainData[:, 2:5],
                       rawtrainData[:, 3:6])  # x3x4,x4x5,x5x6
    arr3 = np.multiply(rawtrainData[:, 0], rawtrainData[:, 0])    # x1^2
    '''
    # numpy : ndarray n-dimensional array.
    # ndarray: scalar, 1-d array(vector), 2-d array(matrix)...n-d array
    # convert 1-d array to 2-d arrayconvert.
    '''
    arr3 = arr3.reshape(len(arr3), 1)
    arr4 = np.multiply(rawtrainData[:, 2], rawtrainData[:, 2])    # x3^2
    # convert vector to matrix
    arr4 = arr4.reshape(len(arr4), 1)
    arr5 = np.ones(np.shape(arr4))                                # b = 1
    trainData = np.concatenate((arr1, arr2, arr3, arr4, arr5), axis=1)
    return trainData

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
    
    trainData = nonlinearTransform(rawtrainData)
    trainLabel = rawtrainLabel
    testData = nonlinearTransform(rawtestData)
    testLabel = rawtestLabel
    
    m, n = np.shape(trainData)
    theta = np.ones(n)
    alpha = 0.5
    maxIteration = 100000
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
    # X = np.arange(100)
    # plt.bar(X,testLabel,color='b',width =0.5)
    # plt.bar(X + 0.5,predictLabel_testData,color='g',width =0.5)
    # plt.show()
    
    # X = np.arange(342)
    # plt.bar(X,trainLabel,color='b',width =0.5)
    # plt.bar(X + 0.5,predictLabel_trainData,color='g',width =0.5)
    # plt.show()


