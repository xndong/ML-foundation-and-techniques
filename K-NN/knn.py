# -*- coding: utf-8 -*-
"""
Created on Wed June 25 21:39:34 2019

@author: DongXiaoning
"""
import numpy as np

def minkowski_distance(x1,x2,distance = "euclidean"):
    if distance =="euclidean":
        sub = np.subtract(x1,x2)
        square = np.dot(sub,sub)
        return np.sqrt(square)
    elif distance =="manhattan":
        sub = np.subtract(x1,x2)
        return np.sum(sub)
    else:
        pass
    return

# yet another generic format, if p =1 ---> manhattan distance, if p=2 euclidean distance....
def minkowski_distance_2(x1,x2,p=2):
    sub = np.subtract(x1,x2)
    identity = np.identity(len(x1))
    diag_matrix = np.matmul(identity,sub)
    power_matrix = np.power(diag_matrix,p)
    ones = np.ones(len(x1))
    product = np.matmul(power_matrix,ones)
    distance = np.power(product,1.0/p)
    return distance







def main():
    nnvalue = 100000.0
    x1 = np.array([10,2,3,4,5])
    x2 = np.array([0,1,2,3,4])

    return

if __name__ == '__main__':
    main()