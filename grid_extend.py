#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:15:19 2021

@author: tangbuxing
"""
import numpy as np
import math
import matplotlib.pyplot as plt


#判断格点场是否为2的n次方大小的正方形,并做延伸

def grid_size(grid):
    X = grid
    #len_lat纬度方向的长度
    len_lat = math.log2(X.shape[0])
    #len_lon经度方向的长度
    len_lon = math.log2(X.shape[1])
    
    if len_lat.is_integer() and len_lon.is_integer():
        print("调用denoise_dwt_2d")
    else:
        #如果纬度方向不满足
        if len_lat.is_integer():
            print("宽的长度满足2的n次方")
        else:
            X_01 = np.full(shape = (abs(X.shape[0] - X.shape[1]), X.shape[1]), fill_value = 0)
            X_x = np.vstack((X, X_01))
        #如果经度方向不满足
        if len_lon.is_integer():
            print("长的长度满足2的n次方")
        else:
            X_01 = np.full(shape = (X.shape[0], abs(X.shape[0] - X.shape[1])), fill_value = 0)
            X_y = np.hstack((X, X_01))    
        #如果纬度、经度方向都不满足
        if not (len_lat.is_integer() and len_lon.is_integer()):
            xy_max = max(math.ceil(len_lat), math.ceil(len_lon))
            X_01 = np.full(shape = (X.shape[0], abs(X.shape[1] - 2**xy_max)), fill_value = 0)
            X_x = np.hstack((X, X_01))
            X_02 = np.full(shape = (abs(X.shape[0] - 2**xy_max), X_x.shape[1]), fill_value = 0)
            X_y = np.vstack((X_x, X_02))    
            res = X_y
        
    return res
        
if __name__ == "__main__":
    x = np.linspace(0,1000,20)
    y = np.linspace(0,500,30)
    X,Y = np.meshgrid(x, y)
    X_Y = grid_size(grid = X)
    plt.imshow(X_Y)
