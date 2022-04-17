# Hackathon Covid 19 project

Our classifier is based on Hackathon covid 19 project. The aim is to develop a easy-to-use covid 19 CT classifier to achieve faster and more accurate diagnosis. 

## Introduction
This study seeks to apply data-quality control techniques and class-sensitive cost functions to enhance the performance of Covid 19 classifier over CT scans of varying serverity. Sliding window is introduced to optimize selection among a group of CT scans of an individual patient. And further DCGan is deployed to reconstruct CT scans. Considering extreme data imbalance of dataset, we maximize the margin among decision boundaries of different classes to boost the performance. We reach the highest MCC of 98% over test set and build up a easy-to-use GUI for future use.

![img](https://lh4.googleusercontent.com/EpIfnvNmK4UT8y3Iy5-4lZ6BGAJMBanp1AQ3K-WOEOR1ASrTbFs0avVLwDnTjT60jF_mfE3hTJZsHpHGfy8WgDFcjt5lNcAZuofc_jaUI1Le7pneEg5nS7Kk29PGaxJNEQCXbognIFPE)

The GUI makes decision based on maxmium votes of different layers from one single nii data sample.

![img](https://lh5.googleusercontent.com/prTGzseYz8uZgRWwTU4R4sKgQJBcH4hHDGcsLoG3PGpCTrMzJ2hV55eCZj2jXzWK7NOasAceSqG3pAfzBw9zqnfbaIO1OmNTpvKkA0SKsqeGoYy-BdQeW69nVEWsnXZxZK8pELIALrun)

## Tips：

1. For every epoch, we will test MCC, F1__score, kappa, balanced_accuracy, precision and recall score over test set. 
2. The model with highest MCC score will be saved in checkpoint directory. 

## Run

- Train: python -u main.py
- Test: python -u test.py
- Run GUI (under gui directory) : 
  - sudo apt install tk-dev python-tk
  - python main.py

## DataSet
* [all dataset](http://storage.yandexcloud.net/covid19.1110/prod/COVID19_1110.zip)

