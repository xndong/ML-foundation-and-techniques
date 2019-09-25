# -*- coding: utf-8 -*-
"""
Created on Wed June 25 21:39:34 2019

@author: DongXiaoning
"""
import numpy as np

def minkowski_distance(x1,x2,distance = "euclid"):
    if distance =="euclid":
        sub = np.subtract(x1,x2)
        square = np.dot(sub,sub)
        return np.sqrt(square)
    elif distance =="manhattan":
        sub = np.subtract(x1,x2)
        return np.sum(sub)
    else:
        pass
    return

def main():
    x1 = np.array([10,2,3,4,5])
    x2 = np.array([0,1,2,3,4])
    print(minkowski_distance(x1,x2))
    return

if __name__ == '__main__':
    main()