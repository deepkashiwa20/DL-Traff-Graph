# [CIKM 2021 Resource Paper] DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction
## DL-Traff-Graph: Graph-Based Deep Learning Models for Urban Traffic Prediction

* Our work has been accepted by CIKM 2021 Resource Track. https://doi.org/10.1145/3459637.3482000
* The preprint version has been uploaded to arXiv. https://arxiv.org/pdf/2108.09091.pdf
* The url of Grid-Based work is : 
(https://github.com/deepkashiwa20/DL-Traff-Grid)

## Cite
@inproceedings{10.1145/3459637.3482000, author = {Jiang, Renhe and Yin, Du and Wang, Zhaonan and Wang, Yizhuo and Deng, Jiewen and Liu, Hangchen and Cai, Zekun and Deng, Jinliang and Song, Xuan and Shibasaki, Ryosuke}, title = {DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction}, year = {2021}, isbn = {9781450384469}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, url = { https://doi.org/10.1145/3459637.3482000 }, doi = {10.1145/3459637.3482000}, booktitle = {Proceedings of the 30th ACM International Conference on Information & Knowledge Management}, pages = {4515–4525}, numpages = {11}, location = {Virtual Event, Queensland, Australia}, series = {CIKM '21}}

## Introduction
English | [简体中文](README_zh-CN.md)

DL-Traff is an open resourse project which offers a benchmark for traffic prediction on grid-based and graph-based models. DL-Traff-Graph is a part of graph-based project. This main branch works on Pytorch1.6. Different versions of Pytorch vary slightly in training time and performance. In this github, we integrate two traditional statistical methods(HistoricalAverage and CopyLastFrame), one time series models (LSTNet) and a large number of graph models into one platform. We maintain that all models are based on the same data processing, the same hyperparameters, and the same computing environment such as the version of Pytorch and Cudnn. Although this makes the models fail to achieve the final convergence effection, the performance of different network architectures under the same conditions will be fully reflected by our experiment. We will update the optimization results of each model in later work.

## Installation Dependencies
Working environment and major dependencies:
* Ubuntu 20.04.2 LTS
* Python 3 (>= 3.6; Anaconda Distribution)
* PyTorch (>= 1.6.0)  py3.6_cuda10.1.243_cudnn7.6.3_0
* torch-summary (>= 1.4.5) <br> you will get some error if you installed torchsummary, see the details at https://pypi.org/project/torch-summary/.<br> please uninstall torchsummary and run ```pip install torch-summary``` to install the new one.
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
  * parameter_STGCN.py  (Model-specific parameter file, which provide the parameters this model will use. If the same parameters as parameter.py appear, this file has priority.)
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
cd /workMETRLA

# Debug the model on video card 1 :
python STGCN.py 1

# Run the main program to train, prediction and test on video card 1:
python pred_STGCN3.py 1

# View the result after the operation is complete.
cd /save/pred_STGCN_METRLA_21061600

```



