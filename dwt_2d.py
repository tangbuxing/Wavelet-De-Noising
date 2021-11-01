# -*- coding: utf-8 -*-
import sys
sys.path.append(r'F:\Work\MODE\tra_test_WaveletDeNoisng')
import math
import pywt
import numpy as np
import pandas as pd
import fun


def dwt_2d(x, wf = "la8", J = 4, method = "universal", H = 0.5, noise_dir = 3, rule = "hard"):
    
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
    
    n = np.size(x)
    #1.分解
    #目前的分解滤波默认为"haar",具体可选小波用pywt.wavelist()查看,小波族查看pywt.families()
    x_dwt = pywt.wavedec2(data = x, wavelet = "haar", level = J)
    
    #2.获取去噪阈值       
    if noise_dir == 3:
        sigma_mad = {"HH":fun.mad(x = x_dwt[4][2]), "HL":fun.mad(x = x_dwt[4][1]), 
                     "LH":fun.mad(x = x_dwt[4][0])}
    else:
        noise = "NULL"    #noise = x_dwt_dict["jj"]暂时不确定什么含义，r里面是为空
        sigma_mad = {"HH":fun.mad(x = noise), "HL":fun.mad(x = noise), 
                     "LH":fun.mad(x = noise)}
    thresh = {"HH": np.repeat(np.sqrt(2*sigma_mad["HH"]**2*np.log(n)), J),
              "HL": np.repeat(np.sqrt(2*sigma_mad["HL"]**2*np.log(n)), J), 
              "LH": np.repeat(np.sqrt(2*sigma_mad["LH"]**2*np.log(n)), J)}
    if method == "long-memory":
        print("默认 method 为 universal")
        
    #3.阈值去噪
    x_dwt_threshold = []
    x_dwt_threshold.append(x_dwt[0])
    for j in range(1,J+1):
        #print(j)
        x_dwt_threshold_1 = []
        if rule == "hard":
            x_dwt_threshold_2 = pywt.threshold(data =  x_dwt[j][0], value = thresh['LH'][j-1],mode = 'hard')
            x_dwt_threshold_1.append(x_dwt_threshold_2)
        else:
            x_dwt_threshold_2 = pywt.threshold(data =  x_dwt[j][0], value = thresh['LH'][j-1], mode = 'soft')
            x_dwt_threshold_1.append(x_dwt_threshold_2)            
        if rule == "hard":
            x_dwt_threshold_2 = pywt.threshold(data =  x_dwt[j][1], value = thresh['HL'][j-1],mode = 'hard')
            x_dwt_threshold_1.append(x_dwt_threshold_2)
        else:
            x_dwt_threshold_2 = pywt.threshold(data =  x_dwt[j][1], value = thresh['HL'][j-1], mode = 'soft')
            x_dwt_threshold_1.append(x_dwt_threshold_2)  
        if rule == "hard":
            x_dwt_threshold_2 = pywt.threshold(data =  x_dwt[j][2], value = thresh['HH'][j-1],mode = 'hard')
            x_dwt_threshold_1.append(x_dwt_threshold_2)
        else:
            x_dwt_threshold_2 = pywt.threshold(data =  x_dwt[j][2], value = thresh['HH'][j-1], mode = 'soft')
            x_dwt_threshold_1.append(x_dwt_threshold_2)  
        x_dwt_threshold.append(tuple(x_dwt_threshold_1))
    
    #4.重构        
    res = pywt.waverec2(coeffs =  x_dwt_threshold, wavelet = "haar")
    
    return res


if __name__ == "__main__":
    data_obs = pd.read_csv(r"G:\Work\MODE\tra_test\FeatureFinder\pert000.csv")
    look_wave = dwt_2d(x = data_obs)
    
    
    
    
    
    
    
    
    