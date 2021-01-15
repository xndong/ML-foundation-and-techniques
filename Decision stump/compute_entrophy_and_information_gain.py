# -*- coding: utf-8 -*-
"""
@author: DongXiaoning
"""
# compute information gain from gini index or entrophy
'''
gini index and entrophy---> measure the impurity of a group.

gini index  ---> information gain          : CART
entrophy    ---> information gain          : ID3
            ---> information gain ration   : C4.5 new to ID3 
'''

import numpy as np
import operator
import collections
import sklearn.datasets

# compute entrophy
def compute_entrophy(group):
    m,n = group.shape
    data = group[:,:-1]
    label = group[:,-1]
    dict_label = collections.Counter(label)
    group_size = float(m)
    if group_size == 0:
        entrophy = 0
    else:
        proportion = np.array(list(dict_label.values()))/group_size

        entrophy = 1 - np.dot(proportion,np.log2(proportion))    ### here is the only difference between entrophy and gini.
    return entrophy

# compute information gain from gini index or entrophy
#
#    information gain = gini(group) - [gini(subgroup_one)* weight + gini(subgroup_two)* weight ]
#
#    weight = subgroup / group
#   
#    weighted gini: gini(subgroup_one)* weight
#
#    gini can be replaced with entrophy
#
# when we split one group into two or more subgroups, we use information gain to describe/measure这个过程中purity or impurity的变化。
def compute_information_gain(entrophy_group,entrophy_subgroup1,weight1,entrophy_subgroup2,weight2):
    return entrophy_group - (entrophy_subgroup1 * weight1 + entrophy_subgroup2 * weight2)


if __name__ == '__main__':
    
  # print(breast_dataset.keys())
  # print(breast_dataset.data.shape) # # When accessing a dictionary, you can use either dictionary[key] or dictionary.key
  # print(breast_dataset.target.shape) # # When accessing a dictionary, you can use either dictionary[key] or dictionary.key

    breast_dataset = sklearn.datasets.load_breast_cancer()
    breast_data = breast_dataset.data
    m,n = breast_data.shape
    breast_label =breast_dataset.target
    breast_label = breast_dataset.target.reshape(m,1)
    group = np.concatenate((breast_data,breast_label),axis = 1)
    entrophy = compute_entrophy(group)

# compute information gain
    entrophy_dict = {}
    info_gain_dict = {}
    for i in range(m):    # 这只是自上而下一行行地split。理论上你可以随意shuffle,split--->所以有 C(m,0) + C(m,1) + C(m,2) +...+ C(m,m-1) + C(m,m)种split
# group1 : entrophy and weight
        group1 = group[:i,:]
        group1_size = float(i)
        entrophy_group1 = compute_entrophy(group1)
        weight_group1 = group1_size / m
# group2 : entrophy and weight
        group2 = group[i:,:]
        group2_size = m - float(i)
        entrophy_group2 = compute_entrophy(group2)
        weight_group2 = group2_size / m
# info gain
        info_gain = compute_information_gain(entrophy,entrophy_group1,weight_group1,entrophy_group2,weight_group2)
        info_gain_dict[i] = info_gain
    largest_info_gain = max(info_gain_dict.items(),key=operator.itemgetter(1))
    print(f'largest info gain is: {largest_info_gain}')
