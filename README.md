# Bilinear_CNN
[![](https://img.shields.io/badge/Bilinear-Model-green.svg)](https://github.com/Ylexx/Bilinear_CNN)

A pytorch implementation of Bilinear CNNs for Fine-grained Visual Recognition(BCNN).

## Requirements
- python 2.7
- pytorch 0.4.1

## Train

Step 1. 
- Download the vgg16 pre-training parameters.
[vgg16-download](https://pan.baidu.com/s/1OkIuKosTRfcZlDXkOW4WLQ). 

- Download the CUB-200-2011 dataset.
[CUB-download](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

Step 2. 
- Set the path to the dataset and vgg parameters in the code.

Step 3. Train the fc layer only. It gives 77.30% test set accuracy.
- python train_last.py


    	


Step 4. Fine-tune all layers. It gives 84.40% test set accuracy.
- python train_finetune.py


	
