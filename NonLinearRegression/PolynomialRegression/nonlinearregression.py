# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:34:26 2019

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
    # convert vector to matrix. ndarray: scalar, vector, matrix
    arr3 = arr3.reshape(len(arr3), 1)
    arr4 = np.multiply(rawtrainData[:, 2], rawtrainData[:, 2])    # x3^2
    # convert vector to matrix
    arr4 = arr4.reshape(len(arr4), 1)
    trainData = np.concatenate((arr1, arr2, arr3, arr4), axis=1)
    return trainData


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
