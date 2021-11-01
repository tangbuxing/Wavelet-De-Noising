# -*- coding: utf-8 -*-
import sys
sys.path.append(r'F:\Work\MODE\tra_test_WaveletDeNoisng')
import math
import pywt
import numpy as np
import pandas as pd
import fun
import dwt_2d


def denoise_dwt_2d(x, wf = "la8", J = 4, method = "universal", H = 0.5, noise_dir = 3, rule = "hard"):
    
    res = {}
    
    #阈值去噪，跟pywt.threshold()mode='soft'/'hard'是一致的
    #'soft':``data/np.abs(data) * np.maximum(np.abs(data) - value, 0)``
    #'hard':一致
    #下面为r的阈值去噪步骤，没调用，仅供对比
    def soft(x, delta):
        #print(delta)
        res_soft = np.sign(x) * np.max(abs(x) - delta, 0)    #np.sign()是取数字符号函数，如果x>0:sign(x) = 1,x=0:sign(x)=0;x < 0:sign(x)=-1
        return res_soft
    
    def hard(x, delta):
        #print(delta)
        if np.any(abs(x) > delta):
            return x
        else:
            return 0
        
    #X = grid
    #len_lat纬度方向的长度
    len_lat = math.log2(x.shape[0])
    #len_lon经度方向的长度
    len_lon = math.log2(x.shape[1])
    
    if len_lat.is_integer() and len_lon.is_integer():            
        res = dwt_2d.dwt_2d(x = x)
    else:
        #如果纬度方向不满足
        if len_lat.is_integer():
            print("宽的长度满足2的n次方")
        else:
            x_extend_01 = np.full(shape = (abs(x.shape[0] - x.shape[1]), x.shape[1]), fill_value = 0)
            x_extend = np.vstack((x, x_extend_01))
            res = dwt_2d.dwt_2d(x = x_extend)
        #如果经度方向不满足
        if len_lon.is_integer():
            print("长的长度满足2的n次方")
        else:
            x_extend_01 = np.full(shape = (x.shape[0], abs(x.shape[0] - x.shape[1])), fill_value = 0)
            x_extend = np.hstack((x, x_extend_01))  
            res = dwt_2d.dwt_2d(x = x_extend)
        #如果纬度、经度方向都不满足
        if not (len_lat.is_integer() and len_lon.is_integer()):
            xy_max = max(math.ceil(len_lat), math.ceil(len_lon))
            x_extend_01 = np.full(shape = (x.shape[0], abs(x.shape[1] - 2**xy_max)), fill_value = 0)
            x_extend_x = np.hstack((x, x_extend_01))
            x_extend_02 = np.full(shape = (abs(x.shape[0] - 2**xy_max), x_extend_x.shape[1]), fill_value = 0)
            x_extend_xy = np.vstack((x_extend_x, x_extend_02))    
            res = dwt_2d.dwt_2d(x = x_extend_xy)
    
    res = res[:x.shape[0], :x.shape[1]]
    
    return res


if __name__ == "__main__":
    data_obs = np.array(pd.read_csv(r"F:\Work\MODE\tra_test\FeatureFinder\pert000.csv"))
    look_wave = denoise_dwt_2d(x = data_obs)
    
    
    
    
    
    
    
    
    