# DL-Traff-Graph: Graph-Based Deep Learning Models for Urban Traffic Prediction

## Introduction
English | [简体中文](README_zh-CN.md)

DL-Traff是一个开放资源项目，为基于网格和基于图形的模型的流量预测提供了基准。DL-Traff-Graph是基于图形的项目的一部分。这部分主要工作在PyTorch1.6上。需要注意的是，不同版本的Pytorch在训练时间和性能上略有不同。在这个github中，我们将两种传统的统计方法、一种时间序列模型和大量的图神经网络模型集成到一个平台中。我们保证了所有模型都基于相同的数据处理、相同的超参数和相同的计算环境，如Pytorch和Cudnn的版本。尽管这会使得各个模型没有达到最终收敛的效果，但是正因如此而可以充分体现不同网络架构在同条件下的表现性能。
## Installation Dependencies
Working environment and major dependencies:
* Ubuntu 20.04.2 LTS
* Python 3 (>= 3.6; Anaconda Distribution)
* PyTorch (>= 1.6.0)  py3.6_cuda10.1.243_cudnn7.6.3_0
* torch-summary (>= 1.4.5) <br> you will get some error if you installed torchsummary, see the details at https://pypi.org/project/torch-summary/.<br> please run ```pip install torch-summary``` to install it.
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
Download this project into your device, the code project will be downloaded into the current path where you type this powershell command:
```
git clone git@github.com:deepkashiwa20/DL-Traff-Graph.git
```

Use the STGCN model on METRLA dataset as an example to demonstrate how to use it. 
* dataset 
<br>**NOTE!** <br> remember to unzip pems-bay.zip to pems-bay.h5 before you run the programe on PEMSBAY dataset, we only upload compressed data due to file size limitation.
```
cd /PEMSBAY
unzip pems-bay.h5
```

* debug and run model
```
cd /workMETR

# Debug the model on video card 1 :
python STGCN.py 1

# Run the main program to train, prediction and test on video card 1:
python pred_STGCN3.py 1

# View the result after the operation is complete.
cd /save/pred_STGCN_METRLA_21061600

```
