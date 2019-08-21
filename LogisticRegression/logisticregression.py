# -*- coding: utf-8 -*-
"""
@author: DongXiaoning
"""
#console: autopep8 --in-place --aggressive --aggressive nonlinearregression.py
#console: yapf -h
# 注意label是{0,1}还是{-1,1}；李航采用前者，林轩田采用后者。hypothesis的函数表达因此略有不同。

import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def getData(dataSet):
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:, :-1] = dataSet[:, :-1]
    trainLabel = dataSet[:, -1]
    for index, element in enumerate(trainLabel):
        if element == 1:
            trainLabel[index] = 0
        else:
            trainLabel[index] = 1
    return trainData, trainLabel


def logisticFunction(theta):  # a.k.a. sigmoid function
    return 1. / (1 + np.exp(-1 * theta))

# using gradient descent as solver (of optimization problem), vector/matrixwise


def gradientDescent(trainData, trainLabel, w, alpha, maxIteration):
    m, n = np.shape(trainData)
    xTrans = trainData.transpose()
    for it in range(maxIteration):
        score = np.dot(trainData, w)
        hypothesis = []
        for i in range(m):
            hypothesis.append(logisticFunction(score[i]))
        error = np.subtract(hypothesis, trainLabel)
        gradient = np.dot(xTrans, error) / m
        w = w - alpha * gradient
        if it % 1000 == 0:
            print("current w: ", w)
    return w, gradient

# using stochastic gradient descent as solver (of optimization problem),
# vector/matrixwise


def stochasticGradientDescent(trainData, trainLabel, w, alpha, maxIteration):
    m, n = np.shape(trainData)
    for it in range(maxIteration):
        index = it % m
        score = np.dot(trainData[index, :], w)
        hypothesis = logisticFunction(score)
        error = hypothesis - trainLabel[index]
        stochastic_gradient = error * trainData[index, :]
        w = w - alpha * stochastic_gradient
        if it % 1000 == 0:
            print("current w: ", w)
    return w, stochastic_gradient

# using batch gradient descent as solver (of optimization problem),
# vector/matrixwise


def batchGradientDescent(
        trainData,
        trainLabel,
        w,
        alpha,
        batchSize,
        maxIteration):
    m, n = np.shape(trainData)
    if m % batchSize == 0:
        totalBatch = m / batchSize
    else:
        totalBatch = m / batchSize + 1
    for it in range(maxIteration):
        currentBatchIndex = it % totalBatch
        if currentBatchIndex == totalBatch - 1:
            # the last batch
            trainData = trainData[currentBatchIndex:
                                  currentBatchIndex + batchSize - 1, :]
            trainLabel = trainLabel[currentBatchIndex:-1]
            xTrans = trainData.transpose()
        else:
            # slicing
            beginIndex = currentBatchIndex * batchSize
            endIndex = (currentBatchIndex + 1) * batchSize - 1
            trainData = trainData[beginIndex:endIndex + batchSize - 1, :]
            trainLabel = trainLabel[currentBatchIndex:endIndex]
            xTrans = trainData.transpose()
        score = np.dot(trainData, w)
        hypothesis = []
        for i in range(m):
            hypothesis.append(logisticFunction(score[i]))
        error = np.subtract(hypothesis, trainLabel)
        batch_gradient = np.dot(xTrans, error) / m
        w = w - alpha * batch_gradient
        if it % 1000 == 0:
            print("current w: ", w)
        return w, batch_gradient

# using gradient descent as solver (of optimization problem), elementwise


def gradientDescent2(trainData, trainLabel, w, alpha, maxIteration):
    m, n = np.shape(trainData)
    for it in range(maxIteration):
        old_w = w
        count = 0
        for j in range(n):
            for i in range(m):
                score = np.dot(trainData[i, :], old_w)
                hypothesis = logisticFunction(score)
                error = hypothesis - trainLabel[i]
                count += error * trainData[i][j]
            w[j] = w[j] - alpha * (1.0 / m) * count
        if it % 1000 == 0:
            print("current w: ", w)
    return w, "gradient is not saved in this function."

# using stochastic gradient descent as solver (of optimization problem),
# elementwise


def stochasticGradientDescent2(trainData, trainLabel, w, alpha, maxIteration):
    m, n = np.shape(trainData)
    for it in range(maxIteration):
        old_w = w
        index = it % m
        for j in range(n):
            score = np.dot(trainData[index, :], old_w)
            hypothesis = logisticFunction(score)
            error = hypothesis - trainLabel[index]
            w[j] = w[j] - alpha * error * trainData[index, j]
        if it % 1000 == 0:
            print("current w: ", w)
    return w, "gradient is not saved in this function."


def predict(data, w):
    m, n = np.shape(data)
    score = np.dot(data, w)
    hypothesis = []
    for i in range(m):
        if logisticFunction(score[i]) >= 0.5:
            hypothesis.append(1)
        else:
            hypothesis.append(0)
    return hypothesis


def countZeroOneError(hypothesis, trainLabel):
    count = 0
    for elementHypo, elementLabel in zip(hypothesis, trainLabel):
        if elementHypo != elementLabel:
            count += 1
    return count


dataPath = r"C:\Users\DongXiaoning\Downloads\Algorithms in Machine Learning Foundation and Techniques\LogisticRegression\heart.dat"
dataSet = genfromtxt(dataPath, delimiter=' ')
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
w = np.ones(n)
alpha = 0.005
maxIteration = 4000000
w, gradient = stochasticGradientDescent2(
    trainData, trainLabel, w, alpha, maxIteration)
hypothesis = predict(trainData, w)
print("\n")
print("Final w's value is: ", w)
print("Final gradient is: ", gradient)
print("Final zero-one error is: ", countZeroOneError(hypothesis, trainLabel))
