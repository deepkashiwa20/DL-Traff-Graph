import os

import numpy as np
import pandas as pd
from _datetime import datetime

# Parameters
WEEK = 2
DAY = 1
HOUR = 2
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
TIMESTEP_PER_HOUR = 12
TIMESTEP_IN = 12
TIMESTEP_OUT = 12

assert TIMESTEP_IN == TIMESTEP_OUT, \
    "ASTGCN just has one timestep parameter Tp, which means TIMESTEP_IN must equal TIMESTEP_OUT."

# ASTGCN just has one timestep parameter Tp for both in and out, which means TIMESTEP_IN must equal TIMESTEP_OUT.
# Thus, in this script we only keep TIMESTEP_OUT as Tp.

# LOAD_PATH : path to load the dataset
# SAVE_DIR : directory to save the prepared data
MODEL_NAME = 'ASTGCN'
DATASET_NAME = 'METRLA'
LOAD_PATH = '../' + DATASET_NAME + '/metr-la.h5'
SAVE_DIR = '../DataTemp/' + DATASET_NAME
ADD_STRING = '_W' + str(WEEK) + '_D' + str(DAY) + '_H' + str(HOUR)
ADD_STRING += '_Official'
INTERVALS = [
    {
        'name': 'WEEK',
        'len': 7 * 24 * TIMESTEP_PER_HOUR,
        'num': WEEK
    },
    {
        'name': 'DAY',
        'len': 24 * TIMESTEP_PER_HOUR,
        'num': DAY
    },
    {
        'name': 'HOUR',
        'len': TIMESTEP_PER_HOUR,
        'num': HOUR
    }
]


# Get samples by interval depend on week,day or hour.
def getXDataByInterval(start_index, interval_len, interval_num, data, x):
    index = [j for j in range(start_index - interval_len * interval_num, start_index, interval_len)]
    sample = np.concatenate([data[k:k + TIMESTEP_OUT] for k in index], axis=0)
    if str(type(x)) == "<class 'NoneType'>":
        x = sample
    else:
        x = np.concatenate([x, sample], axis=0)
    return x


# Build dataset.
def getDataset(data):
    start_index = max(TIMESTEP_PER_HOUR * 24 * 7 * WEEK, 24 * TIMESTEP_PER_HOUR * DAY, TIMESTEP_PER_HOUR * HOUR)
    SAMPLE_NUM = data.shape[0] - TIMESTEP_OUT - start_index + 1
    TRAIN_VAL_SPLIT = int(TRAIN_RATIO * SAMPLE_NUM)
    VAL_TEST_SPLIT = int((TRAIN_RATIO + VAL_RATIO) * SAMPLE_NUM)

    X, Y = [], []
    for i in range(SAMPLE_NUM):
        x = None
        for intervalIter in INTERVALS:
            if intervalIter['num'] != 0:
                x = getXDataByInterval(start_index + i, intervalIter['len'], intervalIter['num'], data, x)
        y = data[i + start_index:i + start_index + TIMESTEP_OUT]
        X.append(x), Y.append(y)

    X, Y = np.array(X), np.array(Y)
    X = X[:, :, np.newaxis, :]
    X = X.transpose(0, 3, 2, 1)
    Y = Y.transpose(0, 2, 1)
    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)

    trainXS, trainYS = X[0:TRAIN_VAL_SPLIT], Y[0:TRAIN_VAL_SPLIT]
    valXS, valYS = X[TRAIN_VAL_SPLIT:VAL_TEST_SPLIT], Y[TRAIN_VAL_SPLIT:VAL_TEST_SPLIT]
    testXS, testYS = X[VAL_TEST_SPLIT:], Y[VAL_TEST_SPLIT:]

    return trainXS, trainYS, valXS, valYS, testXS, testYS


# Print parameters.
def printParam():
    print("\nMODEL_NAME:", MODEL_NAME, "\nDATASET_NAME:", DATASET_NAME)

    print("\nTRAIN_RATIO:", TRAIN_RATIO, "\nVAL_RATIO:", VAL_RATIO, "\nTEST_RATIO:", TEST_RATIO)

    print("\n{:^32}".format("INTERVAL"))
    print("{:<13}{:<13}{:<8}".format("NAME", "LENGTH", "NUM"))
    for intervalIter in INTERVALS:
        print("{:<13}{:<13}{:<8}".format(intervalIter['name'], intervalIter['len'], intervalIter['num']))
    print("\n")

    return


# Load data from file.
def dataloader():
    data_suffix = os.path.splitext(LOAD_PATH)[-1]

    if data_suffix == ".h5":
        data = pd.read_hdf(LOAD_PATH).values
    elif data_suffix == ".npz" or data_suffix == ".npy":
        data = np.load(LOAD_PATH)
        print(data.files)
    elif data_suffix == ".csv":
        data = pd.read_csv(LOAD_PATH)

    print("Dataset is a ", data_suffix, " file.")
    print("Dataset shape: ", data.shape)

    return data


def main():
    saveStr = SAVE_DIR + '/' + MODEL_NAME + ADD_STRING
    
    # mkdir the save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # check if the processed data already exist
    if os.path.exists(saveStr + '.npz'):
        print(saveStr + ' already exist.\nPlease delete the data if necessary (e.g., change the train spatio).')
        return

    data = dataloader()
    printParam()
    trainX, trainY, valX, valY, testX, testY = getDataset(data)
    np.savez_compressed(saveStr, trainX=trainX, trainY=trainY, valX=valX, valY=valY, testX=testX, testY=testY)
    print('Prepared data saved as ' + saveStr + '.')


if __name__ == '__main__':
    main()
