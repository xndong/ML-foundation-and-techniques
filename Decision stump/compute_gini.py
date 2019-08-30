# -*- coding: utf-8 -*-
"""
@author: DongXiaoning
"""
# compute gini index for a group
import sklearn.datasets
import numpy as np
import collections

def compute_gini(group):
    m,n = group.shape
    data = group[:,:-1]
    label = group[:,-1]
    dict_label = collections.Counter(label)
    group_size = float(m)
    if group_size == 0:
        gini_index = 0
    else:
        proportion = np.array(list(dict_label.values()))/group_size
        gini_index = 1 - np.dot(proportion,proportion)
    return gini_index



if __name__ == '__main__':
    dataset = sklearn.datasets.load_breast_cancer()
    data = dataset.data
    m,n = data.shape
    label = dataset.target.reshape(m,1)
    group = np.concatenate((data,label),axis = 1)
    gini = compute_gini(group)
    print(gini)