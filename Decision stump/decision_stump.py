# -*- coding: utf-8 -*-
"""
@author: DongXiaoning
"""

import numpy as np
import operator
import collections
import sklearn.datasets

# compute gini index
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

def compute_information_gain(gini_group,gini_subgroup1,weight1,gini_subgroup2,weight2):
    return gini_group - (gini_subgroup1 * weight1 + gini_subgroup2 * weight2)

def predict(data,stump):
    if data[stump[1]] >= stump[4]:
        return 0
    return 1

if __name__ == '__main__':
    breast_dataset = sklearn.datasets.load_breast_cancer()
    breast_data = breast_dataset.data
    m,n = breast_data.shape
    breast_label =breast_dataset.target
    breast_label = breast_dataset.target.reshape(m,1)    
    group = np.concatenate((breast_data,breast_label),axis = 1)
    
    m,n = group.shape
    gini = compute_gini(group)
# compute info gain
    largest_info_gain_list = []     # on each attributes
    info_gain_dict = {}
    for i in range(n-1):         # traverse each attribute/col
        for j in range(m-1):     # traverse each row
            # split into two groups
            mask = group[:,i] >= group[j][i]    # mask is like a filter, which compares each element in space object
            index = np.where(mask)              # (here is group[:,j]) with group[i][j].
            group1 = group[index]       # index is a tuple and only has an element(size = 1), the element is a list.
            row,col = group1.shape      # thus, group[index,:] will output undesirable result.
            group1_size = float(row)
            mask = group[:,i] < group[j][i]
            index = np.where(mask)
            group2 = group[index]
            row,col = group2.shape
            group2_size = float(row)
            # group1 : gini and weight                  
            gini_group1 = compute_gini(group1)  
            weight_group1 = group1_size / m     
            # group2 : gini and weight
            gini_group2 = compute_gini(group2)
            weight_group2 = group2_size / m
            # info gain
            info_gain = compute_information_gain(gini,gini_group1,weight_group1,gini_group2,weight_group2)
            info_gain_dict[j] = info_gain
        largest_info_gain = max(info_gain_dict.items(),key=operator.itemgetter(1))
        print(f'Attribute {i}\'s name is \'{breast_dataset.feature_names[i]}\', split node is in row {largest_info_gain[0]} ---> value is {group[largest_info_gain[0]][i]}, info gain is: {largest_info_gain[1]}')
        largest_info_gain_list.append((f'attribute {i}',i,breast_dataset.feature_names[i],largest_info_gain[0],group[largest_info_gain[0]][i],largest_info_gain[1]))
    s = max(largest_info_gain_list,key = operator.itemgetter(-1))
    print(f'Best split attribute is \'{s[0]}\' : {s[2]}, and split node is in row {s[3]}, value is {s[4]}')

# add test code to test our result    
    mask = group[:,20] >= 16.82
    index = np.where(mask)
    group3 = group[index]
    mask = group[:,20] < 16.82
    index = np.where(mask)
    group4 = group[index]

