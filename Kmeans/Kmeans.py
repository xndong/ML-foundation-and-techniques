# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:50:07 2019

@author: DongXiaoning
"""

"""
# Location, i.e clustered by the value in this location
# Shape
# Density
"""
import numpy as np
import sklearn
import os.path
import sys

def load_data():
    raw_data_array = np.genfromtxt('cars.csv',delimiter = ',')
    data = raw_data_array[1:,:-1]
    return data

# how many k : no rules.
# initialize k centers: several methods ---> random, distance-based
def init_centers(k,n):
    center_list=[]
    for i in range(k):
        center = 10*np.random.rand(n,) + 0.5
        center_list.append(center)
    return center_list
def rand_sample_as_initcenter(k,data):
    m,n = np.shape(data)
    rows = np.random.randint(0,m,k)
    centers = data[rows]
    return centers
    
# compute distance between center and data point, several distance metrics
def compute_distance(center,point,distance_metric='euclidean'):
    if distance_metric == 'euclidean':
        return euclidean_distance(center,point)
    else:
        return manhattan_distance(center,point)
    
# distance metrics, Euclidean and Manhattan Distance as examples
def euclidean_distance(point_x,point_y):
    vector = np.subtract(point_x,point_y)
    distance = np.dot(vector,vector)
    return distance
    
def manhattan_distance(point_x,point_y):
    vector = np.subtract(point_x,point_y)
    distance = np.sum(vector)
    return distance

# compute which cluster the point should belong to
def choose_cluster():
    return

# Note that when initial cluster number increase, the total cost can usually decrease. Thus, when doing model selection, ensure that k is the same!
# Or give a penalty on k ---> what we usually do.
def total_cost():
    return

def kmeans():
    return

def main():
    data = load_data()
    m,n = np.shape(data)
    center_number = 5
    center_category = ['one','two','three','four','five']
    distance_list = []
    centers = rand_sample_as_initcenter(center_number,data)   # center_list = init_centers(center_number,n)
    for i in range(m):
        for j in range(center_number):
            distance = compute_distance(centers[j],data[i])
            distance_list.append(distance)
        min_distance = min(distance_list)
        index = distance_list.
    return

if __name__ == "__main__":
    main()