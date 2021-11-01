# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:34:06 2021
"""
import numpy as np
import pandas as pd

def thresholder(x, th, Type = ['binary', 'replace_below'], rule = ">=", replace_with = 0.0):
    if rule == ">=":
        Id = x>=th
    elif rule == ">":
        Id = x>th
    elif rule == "<=":
        Id = x<=th
    elif rule == "<":
        Id = x<th    
    xdim = x.shape
    if xdim == None:
        out = np.full(len(x), replace_with)
    else:
        out = np.full(xdim, replace_with)
    if Type == "binary":
        out[Id] = 1
    else: 
        out[Id] = x[Id]
    return out

'''
if __name__ == "__main__":
    data_obs = pd.read_csv(r"F:\Work\MODE\tra_test\FeatureFinder\pert000.csv")
    out_thresholder = thresholder(x = np.array(data_obs), th = 1.016)    #x为格点场, th为阈值
'''