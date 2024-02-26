# Multi-label classification of CXR-14 dataset

This repository contains the code for the classification of the CXR-14 dataset. The dataset contains 14 different classes of chest X-ray images. The dataset is available at [Kaggle](https://www.kaggle.com/nih-chest-xrays/data).
Have tested using DenseNet121 from torchxrayvision library. 

To run the code do `bash train.sh` while on the idun cluster. This script will create a slurm job and run the code on the cluster.

## Info

Did not move further with this because multi-label classification is hard enough as is, will look further into single classification, and maybe pick up this later.