# -*- coding: utf-8 -*-
"""
@author: Dong Xiaoning
"""
# compute information gain from gini index or entrophy
'''
 gini index and entrophy---> measure the impurity of a group.

 In two or more groups(eg a dataset is splited into two groups),
 we weightly add gini index from coresponding group together to compute information gain.
'''

import numpy as np
import operator
import collections
import sklearn.datasets

def compute_gini(data,label,row_index):
    m,n = data.shape
    assert row_index <= m and row_index >= 0,'row index is out of boundary'
    group1 = label[:row_index]
    group2 = label[row_index:]
    dict_group1 = collections.Counter(group1)
    dict_group2 = collections.Counter(group2)
    group1_size = float(row_index)
    group2_size = m - float(row_index)
# group1's gini and weight
    if group1_size == 0:
        gini_group1 = 0
    else:
        proportion1 = np.array(list(dict_group1.values()))/group1_size
        gini_group1 = 1 - np.dot(proportion1,proportion1)
    weight_group1 = group1_size / m
# group2's gini and weight
    if group2_size == 0:
        gini_group2 = 0
    else:
        proportion2 = np.array(list(dict_group2.values()))/group2_size
        gini_group2 = 1 - np.dot(proportion2,proportion2)
    weight_group2 = group2_size / m
# total gini of two groups
    gini_index = gini_group1 * weight_group1 + gini_group2 * weight_group2
    return gini_index

def compute_gini_2(data,label,row_index):
    m,n = data.shape
    assert row_index <= m and row_index >= 0,'row index is out of boundary'
    categories = list(set(label))     
    group1 = label[:row_index]
    group2 = label[row_index:]
    group1_size = float(row_index)
    group2_size = m - float(row_index)
# group1' gini
    if group1_size == 0:
        gini_group1 = 0
    else:
        proportion = []
        for category in categories:
            mask = group1 == category
            count, = group1[mask].shape
            proportion.append(count/group1_size)
        proportion = np.array(proportion)
        gini_group1 = 1 - np.dot(proportion,proportion)
    weight_group1 = group1_size / m
# group2's gini and weight
    if group2_size == 0:
        gini_group2 = 0
    else:
        proportion = []
        for category in categories:
            mask = group2 == category
            count, = group2[mask].shape
            proportion.append(count/group2_size)
        proportion = np.array(proportion)
        gini_group2 = 1 - np.dot(proportion,proportion)
    weight_group2 = group2_size / m
# total gini of two groups and weight
    gini_index = gini_group1 * weight_group1 + gini_group2 * weight_group2
    return gini_index


# compute information gain from gini index or entrophy
#
#    information gain = gini(group) - [gini(subgroup_one)* weight + gini(subgroup_two)* weight ]
#
#    weight = subgroup / group
#   
#    weighted gini: gini(subgroup_one)* weight
#
# when we split one group into two or more subgroups, we use information gain to describe/measure这个过程中purity or impurity的变化。
def compute_information_gain(gini_group,gini_two_groups):
    return gini_group - gini_two_groups


if __name__ == '__main__':
    breast_dataset = sklearn.datasets.load_breast_cancer()
    breast_data = breast_dataset.data
    m,n = breast_data.shape
    breast_label =breast_dataset.target

# compute gini index of group
    dict_label = collections.Counter(breast_label)
    proportion = np.array(list(dict_label.values())) / m
    gini_group = 1 - np.dot(proportion,proportion)
# compute gini index of splited data(i.e. two groups)
    gini_index_dict = {}
    for i in range(m):    # 这只是自上而下一行行地split。理论上你可以随意shuffle,split--->所以有 C(m,0) + C(m,1) + C(m,2) +...+ C(m,m-1) + C(m,m)种split
        gini_index = compute_gini(breast_data,breast_label,i)
        gini_index_dict[i] = gini_index
    least_gini_index,least_gini_value = min(gini_index_dict.items(),key=operator.itemgetter(1))
    largest_info_gain = gini_group - least_gini_value
    print(f'largest info gain is: {largest_info_gain}, index is: {least_gini_index}')
