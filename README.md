# DL-Traff-Graph: Graph-Based Deep Learning Models for Urban Traffic Prediction

Papers should include a description of the resource, an illustration of the use case(s) for the resource, explain its utility, reusability.

## Introduction
English | [简体中文](README_zh-CN.md)

DL-Traff is an open resourse project which offers a benchmark for traffic prediction on grid-based and graph-based models. DL-Traff-Graph is a part of graph-based project. This main branch works with Pytorch1.6. Different versions of Pytorch vary slightly in training time and performance. In this github, we integrate two traditional statistical methods, one time series models and a large number of graph models into one platform. We maintain that all models are based on the same data processing, the same hyperparameters, and the same computing environment such as the version of Pytorch and Cudnn.

## Installation Dependencies
Working environment and major dependencies:
* Ubuntu 20.04.2 LTS
* Python 3 (>= 3.6; Anaconda Distribution)
* PyTorch (>= 1.6.0)  py3.6_cuda10.1.243_cudnn7.6.3_0
* torch-summary (>= 1.4.5) <br> you will get some error if you installed torchsummary <br> please run ```pip install torch-summary``` to install it <br> see the details at https://pypi.org/project/torch-summary/
* tables
* pandas
* scipy
* scikit-learn

## Public data and models zoo
### Datasets
* METR-LA
* PeMS-BAY
* PeMSD7(M)

### Models
* HistoricalAverage
* CopyLastSteps
* STGCN
* DCRNN
* GraphWaveNet
* ASTGCN
* GMAN
* MTGNN
* AGCRN
* LSTNet
* TGCN *Need thousands of epochs to converge.

## Components and user guide

### Content
* METR-LA  (dataset folder)
  * metr-la.h5  (feature file)
  * adj_mx.pkl  (asymmetry road adjecent file)
  * W_metrla.csv  (symmetry road adjecent file)
  * ...
* PeMS-BAY
  * pems-bay.h5   (feature file)
  * adj_mx_bay.pkl (asymmetry road adjecent file)
  * W_pemsbay.csv  (symmetry road adjecent file)
  * ...
* PeMSD7(M)
  * V_228.csv  (feature file)
  * W_228.csv  (symmetry road adjecent file)
  * ...
* save  (log and result folder, it will be construct by the program automatically to save train process log and test results)
  * pred_METR-LA_STGCN_2106160000  (a folder named in oder of dataset, model and time of program execution)    
  * pred_METR-LA_DCRNN_2106160000
  * ...
* workMETRLA  (main program folder in METRLA dataset)
  * parameter.py  (common parameter file, which provide the parameters every model will use)
  * parameter_STGCN.py  (model parameter file, which provide the parameters every model will use)
  * STGCN.py  (model file, used for debug and providing model interfaces for pred programs.)
  * pred_STGCN3.csv (pred file, used for train, prediction and test of the single model STGCN)
  * ...
* workPEMSBAY  (main program folder in PEMSBAY dataset)
  * parameter.py  (common parameter file, which provide the parameters every model will use)
  * parameter_STGCN.py  (model parameter file, which provide the parameters every model will use)
  * STGCN.py  (model file, used for debug and providing model interfaces for pred programs.)
  * pred_STGCN3.csv (pred file, used for train, prediction and test of the single model STGCN)
  * ...
* workPEMSD7M  (main program folder in PEMSD7M dataset)
  * parameter.py  (common parameter file, which provide the parameters every model will use)
  * parameter_STGCN.py  (model parameter file, which provide the parameters every model will use)
  * STGCN.py  (model file, used for debug and providing model interfaces for pred programs.)
  * pred_STGCN3.csv (pred file, used for train, prediction and test of the single model STGCN)
  * ...
### User guide
Use the STGCN model on METRLA dataset as an example to demonstrate how to use it. 
* dataset <br> put the prepared dataset into dataset folder <br> Note！ <br> remember to unzip pems-bay.zip to pems-bay.h5 before you run the programe on PEMSBAY dataset, we only upload compressed data due to file size limitation.

* debug and run model
```
cd /workMETR

# Debug the model on video card number 1 :
python STGCN.py 1

# Run the main program to train, prediction and test on video card number 1:
python pred_STGCN3.py 1

# View the result after the operation is complete.
cd /save/pred_STGCN_METRLA_21061600

```



