import pandas as pd
import scipy.sparse as ss
import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
import Metrics
from Param import *
from Param_HistoricalAverage import *

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':
        # Be careful here, start with 7*DAYTTIMESTEP for this method.
        for i in range(7*DAYTIMESTEP, TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def getXSWeek(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    YS, YS_index = [], []
    if mode == 'TRAIN':  
        for i in range(HISTORYDAY*DAYTIMESTEP, TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            YS_index.append(np.arange(i+TIMESTEP_IN, i+TIMESTEP_IN+TIMESTEP_OUT))
            YS.append(data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :])
        YS_index, YS = np.array(YS_index), np.array(YS)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            YS_index.append(np.arange(i+TIMESTEP_IN, i+TIMESTEP_IN+TIMESTEP_OUT))
            YS.append(data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :])
        YS_index, YS = np.array(YS_index), np.array(YS)
    
    XS_Week = []
    for i in range(YS_index.shape[0]):
        Week = []
        for j in range(YS_index.shape[1]):
            index = YS_index[i, j]
            Week.append(data[index-HISTORYDAY*DAYTIMESTEP:index:DAYTIMESTEP, :])
        XS_Week.append(Week)
    XS_Week = np.array(XS_Week)
    return XS_Week, YS

def HistoricalAverage(data, mode):
    XS_Week, YS = getXSWeek(data, mode)
    YS_pred = np.mean(XS_Week, axis=2)
    return YS_pred

def testModel(name, data, mode):
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS, YS = getXSYS(data, mode)
    YS_pred = HistoricalAverage(data, mode)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()

################# Parameter Setting #######################
MODELNAME = 'HistoricalAverage'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
################# Parameter Setting #######################
data = pd.read_hdf(FLOWPATH).values
print('data.shape', data.shape)
###########################################################

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)

    print(KEYWORD, 'testing started', time.ctime())
    testModel(MODELNAME, data, 'TEST')


if __name__ == '__main__':
    main()