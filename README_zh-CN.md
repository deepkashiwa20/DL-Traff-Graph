# DL-Traff-Graph: Graph-Based Deep Learning Models for Urban Traffic Prediction

## Introduction
[English](README.md) | 简体中文

DL-Traff是一个开放资源项目，为基于网格和基于图形的模型的流量预测提供了基准。DL-Traff-Graph是基于图形的项目的一部分。这部分主要工作在PyTorch1.6上。需要注意的是，不同版本的Pytorch在训练时间和性能上略有不同。在这个github中，我们将两种传统的统计方法、一种时间序列模型和大量的图神经网络模型集成到一个平台中。我们保证了所有模型都基于相同的数据处理、相同的超参数和相同的计算环境，如Pytorch和Cudnn的版本。尽管这会使得各个模型没有达到最终收敛的效果，但是正因如此而可以充分体现不同网络架构在同条件下的表现性能。我们会在后续工作中更新各个模型调优后的结果。
## 安装依赖环境
工作环境和主要依赖包:
* Ubuntu 20.04.2 LTS
* Python 3 (>= 3.6; Anaconda Distribution)
* PyTorch (>= 1.6.0)  py3.6_cuda10.1.243_cudnn7.6.3_0
* torch-summary (>= 1.4.5) <br> 如果你安装的是torchsummary，你可能会碰倒报错, 关注源包网址获取更多细节： https://pypi.org/project/torch-summary/.<br> 在删除torchsummary后安装新库： ```pip install torch-summary```
* tables
* pandas
* scipy
* scikit-learn

## 公开数据集和模型库
### 数据集
* METR-LA
* PeMS-BAY
* PeMSD7(M)

### 模型
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

## 组成介绍和用户指导

### 目录
* METR-LA  (数据集文件夹)
  * metr-la.h5  (特征数据文件)
  * adj_mx.pkl  (非对称路网矩阵文件)
  * W_metrla.csv  (对称路网矩阵文件)
  * ...
* PeMS-BAY
  * pems-bay.h5  (特征数据文件)
  * adj_mx_bay.pkl  (非对称路网矩阵文件)
  * W_pemsbay.csv  (对称路网矩阵文件)
  * ...
* PeMSD7(M)
  * V_228.csv  (特征数据文件)
  * W_228.csv  (对称路网矩阵文件)
  * ...
* save  (记录和结果文件夹, 将会被程序自动创建，其中会包含训练，预测和测试的过程和结果记录)
  * pred_METR-LA_STGCN_2106160000  (记录和结果文件命名由 pred+数据+模型+时间的方式组成)    
  * pred_METR-LA_DCRNN_2106160000
  * ...
* workMETRLA  (METRLA 数据集下的主程序)
  * parameter.py  (共同参数文件, 提供每个模型都会用到的参数)
  * parameter_STGCN.py  (模型独有参数文件,提供仅限于本模型的参数。如果出现了和parameter.py一样的参数，本文件的参数将有优先权。)
  * STGCN.py  (模型文件, 用来debug以及提供模型网络给主程序调用)
  * pred_STGCN3.csv (主程序, 针对STGCN网络的训练预测测试文件)
  * ...
* workPEMSBAY  (PEMSBAY 数据集下的主程序)
  * parameter.py  (共同参数文件, 提供每个模型都会用到的参数)
  * parameter_STGCN.py  (模型独有参数文件,提供仅限于本模型的参数。如果出现了和parameter.py一样的参数，本文件的参数将有优先权。)
  * STGCN.py  (模型文件, 用来debug以及提供模型网络给主程序调用)
  * pred_STGCN3.csv (主程序, 针对STGCN网络的训练预测测试文件)
  * ...
* workPEMSD7M  (PEMSD7M 数据集下的主程序)
  * parameter.py  (共同参数文件, 提供每个模型都会用到的参数)
  * parameter_STGCN.py  (模型独有参数文件,提供仅限于本模型的参数。如果出现了和parameter.py一样的参数，本文件的参数将有优先权。)
  * STGCN.py  (模型文件, 用来debug以及提供模型网络给主程序调用)
  * pred_STGCN3.csv (主程序, 针对STGCN网络的训练预测测试文件)
  * ...
### 用户指导
下载源码到你的设备上, 当你进入到一个路径后输入一下指令，代码将会被下载到该路径下:
```
git clone git@github.com:deepkashiwa20/DL-Traff-Graph.git
```

用STGCN在METRLA数据集下的运行来示范使用方法：
* dataset 
<br>**注意！** <br>在你运行PEMSBAY数据集下的程序之前，记得解压pems-bay.zip释放出pems-bay.h5。由于github的文件大小限制，我们只上传的了PEMSBAY的压缩数据文件。
```
cd /PEMSBAY
unzip pems-bay.h5
```

* debug和运行模型
```
cd /workMETR

# 在1号显卡上debug模型 :
python STGCN.py 1

# 在1号显卡是运行主程序进行训练，预测和测试。
python pred_STGCN3.py 1

# 当训练完成后查看结果.
cd /save/pred_STGCN_METRLA_21061600

```
