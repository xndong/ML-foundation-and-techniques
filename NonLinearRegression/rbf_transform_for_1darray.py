# -*- coding: utf-8 -*-
"""
@author: Dong Xiaoning
"""

import numpy as np

def compute_RBF(x, center_numbers, sigma):
    dim = np.shape(x)[0]
    dim, = np.shape(x)  # sequence unpack 序列解包 only one element ---> ','
    centers = np.random.rand(center_numbers, dim)  # 随机选取center_numbers个center
    sigmas = np.array([sigma for i in range(center_numbers)])
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

if __name__ == '__main__':

    x, y, z = np.random.rand(3, 20)

    print(compute_RBF(x,8,1))
    print(compute_RBF(y,8,0.5))
    print(compute_RBF(z,8,2))
    print(compute_RBF(z,8,5))
