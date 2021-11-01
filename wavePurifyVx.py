import meteva.base as meb
import numpy as np
import pandas as pd
import math
import time
import sys
sys.path.append(r'F:\Work\MODE\Submit')
sys.path.append(r'F:\Work\MODE\tra_test_WaveletDeNoisng')
from mode import data_pre
import thresholder, vxstats, denoise_dwt_2d



def wavePurifyVx_default(grd_ob, grd_fo, climate = 'NULL', which_stats = ["bias", "ts", "ets", "pod", "far", "f", "hk", "mse"], 
                         thresholds = 'NULL', rule = ">=", return_fields = True, show = False):
    if show:
        begin_tiid = time.time()
        print(begin_tiid)
    if show:
        print(thresholds)

    #暂时先进行直接赋值
    out = {}
    X = np.squeeze(np.array(grd_ob))
    Xhat = np.squeeze(np.array(grd_fo))
    xdim = np.array(X.shape)

    if any(xdim != Xhat.shape):
        try:
            sys.exit(0)
        except:
            print("wavePurifyVx: dimensions of verification and modelfield must be the same.")
    elementExist = []
    for i, value in enumerate(which_stats):
        elementExist.append(value in ['bias', 'ts', 'ets', 'pod', 'far', 'f', 'hk', 'mse'])
    if thresholds == 'NULL' and any(elementExist):
        thresholds_Xhat = np.quantile(Xhat, [0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95])
        thresholds_X = np.quantile(X, [0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95])
        thresholds = np.vstack((thresholds_Xhat, thresholds_X)).T
        thresholds = pd.DataFrame(thresholds,columns=('Forecast','Verification'))
        #attr(out, "qs") <- c("0", "0.1", "0.25", "0.33", "0.5", "0.66", "0.75", "0.9", "0.95")    #作为一个属性输出
    else:
        if type(thresholds) is list and len(thresholds) != 2:
            thresholds = np.repeat(np.array(thresholds), 2).reshape(len(thresholds), 2)
        if type(thresholds) is not list :
            try:
                sys.exit(0)
            except:
                print("wavePurifyVx: invalid thresholds argument.  Must be a list object.")
        thresholds = pd.DataFrame(thresholds,columns=('Forecast','Verification'))
        out['thresholds'] = thresholds
    out['thresholds'] = thresholds
    args = {'J':'NULL'}

    isnt_J = args['J'] == 'NULL'
    if isnt_J:
        J = math.floor(math.log2(min(xdim)))    #math.floor()为向下取整
        args['J'] = J
    else:
        J = args['J']
    out['args'] = args

    value1 = []
    value2 = []
    for val in xdim:
        value1.append(math.floor(math.log2(val)))    #math.floor()为向下取整
        value2.append(math.ceil(math.log2(val)))    #math.ceil()为向上取整
                
    if value1 == value2:     #通过对格点场的大小做向上和向下取整, 判断格点场是否为2的n次方大小
        dyadic = True
    else:
        dyadic = False
    if show:
        print("\nDenoising the fields.\n")

    #去噪处理,需要调用denoise.modwt.2d/denoise.dwt.2d两个函数
    if dyadic:
        if isnt_J:
            print("'denoise_dwt_2d'不适用于非2的n次方的格点场,此处已做延伸")
            Z = denoise_dwt_2d.denoise_dwt_2d(x = X, J = J)
            #对负值和精度进行控制
            Z = np.abs(np.round(Z, 5))
            Y = denoise_dwt_2d.denoise_dwt_2d(x = Xhat, J = J)
            Y = np.abs(np.round(Y, 5))
            if not climate == 'NULL':
                Climate = denoise_dwt_2d.denoise_dwt_2d(x = climate, J = J)
        else:
            print("'denoise_dwt_2d'不适用于非2的n次方的格点场")
            Z = denoise_dwt_2d.denoise_dwt_2d(x = X)
            Z = np.abs(np.round(Z, 5))
            Y = denoise_dwt_2d.denoise_dwt_2d(x = Xhat)
            Y = np.abs(np.round(Y, 5))
            if not climate == 'NULL':
                Climate = denoise_dwt_2d.denoise_dwt_2d(x = climate)
    else:
        if isnt_J:
            print("'denoise_modwt_2d'适用于非2的n次方的格点场，此处调用已做延伸的'denoise_dwt_2d'")
            Z = denoise_dwt_2d.denoise_dwt_2d(x = X, J = J)
            Z = np.abs(np.round(Z, 5))
            Y = denoise_dwt_2d.denoise_dwt_2d(x = Xhat, J = J)
            Y = np.abs(np.round(Y, 5))
            if not climate == 'NULL':
                Climate = denoise_dwt_2d.denoise_dwt_2d(x = climate, J = J)
        else:
            print("'denoise_modwt_2d'适用于非2的n次方的格点场")
            Z = denoise_dwt_2d.denoise_dwt_2d(x = X)
            Z = np.abs(np.round(Z, 5))            
            Y = denoise_dwt_2d.denoise_dwt_2d(x = Xhat)
            Y = np.abs(np.round(Y, 5))
            if not climate == 'NULL':
                Climate = denoise_dwt_2d.denoise_dwt_2d(x = climate)            
                
    if return_fields:
        out.update({'X': X, 'Xhat':Xhat, 'X_denoised':Z, 'Xhat_denoised':Y})
        if not climate == 'NULL':
            out.update({'Climate':climate, 'Climate2':Climate})

    if any(thresholds != 'NULL'):
        q = thresholds.shape[0]
    else:
        q = 1

    if 'bias' in which_stats:
        out['bias'] = np.full(shape = q, fill_value = np.nan)
    if 'ts' in which_stats:
        out['ts'] = np.full(shape = q, fill_value = np.nan)
    if 'ets' in which_stats:
        out['ets'] = np.full(shape = q, fill_value = np.nan)
    if 'pod' in which_stats:
        out['pod'] = np.full(shape = q, fill_value = np.nan)
    if 'far' in which_stats:
        out['far'] = np.full(shape = q, fill_value = np.nan)
    if 'f' in which_stats:
        out['f'] = np.full(shape = q, fill_value = np.nan)
    if 'hk' in which_stats:
        out['hk'] = np.full(shape = q, fill_value = np.nan)
    if 'mse' in which_stats:
        out['mse'] = np.full(shape = q, fill_value = np.nan)
    if climate != 'NULL':
        out['acc'] = np.full(shape = q, fill_value = np.nan)
    if show:
        print("\nLooping through thresholds = {}\n".format(thresholds))

    out2 = []
    out_bias = []
    out_ts = []
    out_ets = []
    out_pod = []
    out_far = []
    out_f = []
    out_hk = []
    out_acc = []
    for threshold in range(q):
        if show:
            print(threshold)
        if 'mse' in which_stats: 
            if all(thresholds[threshold:threshold+1] != 'NULL'):
                X2 = thresholder.thresholder(x = Z, th = thresholds['Verification'][threshold], Type = 'replace_below', rule = rule)
                Y2 = thresholder.thresholder(x = Y, th = thresholds['Forecast'][threshold], Type = 'replace_below', rule = rule)
                out1 = vxstats.vxstats(X = Y2, Xhat = X2, which_stats = "mse")['mse']
                out2.append(out1)
                #out.update({'%d'%i:out1})
                out['mse'] = out2
            else:
                out1 = vxstats.vxstats(X = Y, Xhat = Z, which_stats = "mse")['mse']
                out2.append(out1)    
                out['mse'] = out2
        elementExist2 = []
        for i, value in enumerate(['bias', 'ts', 'ets', 'pod', 'far', 'f', "hk"]):
            elementExist2.append(value in which_stats)
        if any(elementExist2):
            Xbin = thresholder.thresholder(x = Z, Type = "binary", th = thresholds['Verification'][threshold], rule = rule)
            Ybin = thresholder.thresholder(x = Y, Type = "binary", th = thresholds['Forecast'][threshold], rule = rule)
            if (threshold == 1):
                dostats = np.array(which_stats)
                if ("mse" in dostats):
                    dostats = dostats[dostats != "mse"]
            #tmp = vxstats.vxstats(X = Ybin, Xhat = Xbin, which_stats = dostats.tolist())
            tmp = vxstats.vxstats(X = Ybin, Xhat = Xbin)
            if 'bias' in which_stats:
                out_bias1 = tmp['bias']
                out_bias.append(out_bias1)
                out['bias'] = out_bias
            if 'ts' in which_stats:
                out_ts1 = tmp['ts']
                out_ts.append(out_ts1)
                out['ts'] = out_ts
            if 'ets' in which_stats:
                out_ets1 = tmp['ets']
                out_ets.append(out_ets1)
                out['ets'] = out_ets
            if 'pod' in which_stats:
                out_pod1 = tmp['pod']
                out_pod.append(out_pod1)
                out['pod'] = out_pod
            if 'far' in which_stats:
                out_far1 = tmp['far']
                out_far.append(out_far1)
                out['far'] = out_far
            if 'f' in which_stats:
                out_f1 = tmp['f']
                out_f.append(out_f1)
                out['f'] = out_f               
            if 'hk' in which_stats:
                out_hk1 = tmp['hk']
                out_hk.append(out_hk1)
                out['hk'] = out_hk
        if not climate == 'NULL':
            if not thresholds == 'NULL':
                X2 = thresholder.thresholder(x = Z, Type = "replace_below", th = thresholds['Verification'][threshold], rule = rule)
                Y2 = thresholder.thresholder(x = Y, Type = "replace_below", th = thresholds['Forecast'][threshold], rule = rule)
                Clim = thresholder(x = Climate, Type = "replace_below", th = thresholds['Forecast'][threshold], rule = rule)
                denom = math.sqrt(np.sum(np.sum((Y2 - Clim)**2, axis = 0)))*math.sqrt(np.sum(np.sum((X2 - Clim)**2, axis = 0)))
                #先做矩阵乘法，再获取对角线的值，再求和
                #np.diag为对角线，np.dot()为矩阵乘法
                numer = np.sum(np.diag(np.dot((Y2 - Clim).T, (X2 - Clim).T)))
            else:
                denom = math.sqrt(np.sum(np.sum((Y - Clim)**2, axis = 0)))*math.sqrt(np.sum(np.sum((Z - Clim)**2, axis = 0)))
                numer = np.sum(np.diag(np.dot((Y - Clim).T, (Z - Clim).T)))
            out_acc1 = numer/denom
            out_acc.append(out_acc1)
            out['acc'] = out_acc    
    
    
    if show:
        print(time.time() - begin_tiid)    
        
    return out

if __name__ == '__main__':
    filename_ob = r'G:\\Work\\MODE\\Submit\\mode_data\\ob\\rain03\\20070111.000.nc'    #i = 0, j = 27
    filename_fo = r'G:\\Work\\MODE\\Submit\\mode_data\\ec\\rain03\\20070108.003.nc'
    grd_ob = meb.read_griddata_from_nc(filename_ob)
    grd_fo = meb.read_griddata_from_nc(filename_fo)
    #data_obs = np.array(pd.read_csv(r"F:\Work\MODE\tra_test\FeatureFinder\pert000.csv"))
    #data_fcst = np.array(pd.read_csv(r"F:\Work\MODE\tra_test\FeatureFinder\pert004.csv"))
    #data_loc = pd.read_csv(r"F:\Work\MODE\tra_test\FeatureFinder\ICPg240Locs.csv")
    look_wavePurifyVx = wavePurifyVx_default(grd_ob=grd_ob, grd_fo=grd_fo)

    '''    
    climate = 'NULL'
    which_stats = ["bias", "ts", "ets", "pod", "far", "f", "hk", "mse"]
    thresholds = 'NULL'
    rule = '>='
    return_fields = True
    verbose = False
    time_point = 1
    obs = 1
    model = 1
    show = True
    '''