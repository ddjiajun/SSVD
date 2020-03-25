# SSVD
Implementation of "Single Shot Video Object Detector". The original version is implemented with MXNet, but we convert it into pytorch and take the off-the-shelf library from maskrcnn-benchmark to make it easier to try.

## Installation
See INSTALL.md

## Start
### 1. Dataset preparation
Download the ILSVRC DET and ILSVRC VID datasets, unzip and move them to "./dataset". The structure of dataset looks like:
    
    ./dataset/ILSVRC/
    ./dataset/ILSVRC/Annotations/DET
    ./dataset/ILSVRC/Annotations/VID
    ./dataset/ILSVRC/Data/DET
    ./dataset/ILSVRC/Data/VID
    ./dataset/ILSVRC/ImageSets


### 2. Model downloading
Download our model from 
https://drive.google.com/file/d/1vJkGGgCO-NmzzJmV5vwFisw_3D8VJ_L4/view?usp=sharing.

Create a folder named "ckpt", and put our model into it.

### 3. Setup
    python setup.py build develop

### 4.Test on ILSVRC VID dataset
    bash ssvd_test.sh