# ML with Go

This material introduces some Go packages and frameworks that will help us implement ML in Go. Once you are done with this material, you will know where to look to find ML-related packages for Go, have some hands-on experience working with these packages, and understand the utility of Go for ML.

This section is meant to be a survey of Go packages and frameworks related to ML along with some hands-on exercises using those packages and frameworks. We will be referring back to the ML workflow introduced in [the last section](../ml_workflow) as necessary.  This guide will walk you through:

1. [Data loading and gathering](#1.-data-loading-and-gathering)
2. [Data analysis and visualisation](#2.-data-analysis-and-visualisation)
3. [Classification and clustering](#3.-classification-and-clustering)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Data loading and gathering

We are going to start exploring some JSON data. The dataset comes from the Emoji Similarity Baseline Dataset, with 508 Emoji Pairs and Similarity Ratings. From https://www.kaggle.com/sanjayaw/emosim508
The objective is to start using Jupyter notebooks and exploring data.

Then we are going to use the War Of The Five Kings dataset.
Part of the strength of the models that we build is in the data that we feed them. So we have to choose the good type of data for the right type of model. 
This second part of the notebook is based in the analysis made in https://github.com/chrisalbon/war_of_the_five_kings_dataset, and is based on the dataset of the battles in the War of the Five Kings from George R.R. Martin's A Song Of Ice And Fire series.


## 2. Data analysis and visualisation

To make correct decisions, you have to analyse your Data. This analysis use Datasets. It is important a fundamental understanding of probability and statistics to understand your Data.
In this notebook, we are going to analyse the data of weights and heights of players of the NBL.


## 3. Classification and clustering

There is not only neural networks that can solve complex classification problems or clustering.
We are going to explore a dataset with and unsupervised algorithme called K-means, where we try to figure out which are the bests clusters in our dataset and find a cluster model.
We are going to use the "fleet_data" dataset found as an example in the book [Machine Learning with Go](#resources)

## Resources

- [Machine Learning with Go](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go)  

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
