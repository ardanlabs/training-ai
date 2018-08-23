# ML with Go

This material introduces some Go packages and frameworks that will help us implement ML in Go. Once you are done with this material, you will know where to look to find ML-related packages for Go, have some hands-on experience working with these packages, and understand the various types of machine learning models.

This section is meant to be a survey of Go packages and frameworks related to ML along with some hands-on exercises using those packages and frameworks. We will be referring back to the ML workflow introduced in [the last section](../ml_workflow) as necessary.  This guide will walk you through:

1. [Gathering, Profiling, and Cleaning Data](#1-gathering-profiling-and-cleaning-data)
2. [Defining, Training, and Testing Models](#2-defining-training-and-testing-models)
3. [(Bonus) More sophisticated models](#3-bonus-more-sophisticated-models)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Gathering, Profiling, and Cleaning Data

In this section, we will look at how we can import, parse, manipulate, and profile data with Go. Note, there are innumerable types and formats of data that you might have to deal with in an ML/AI workflow (CSV, JSON, Parquet, Avro,  etc.), and we won't cover all of them. Rather, we will highlight a few of the main Go packages that you can utilize for data gathering,  profiling, and cleaning.  We will look at two different example data sets using Jupyter: (i) an [emoji data set](https://www.kaggle.com/sanjayaw/emosim508) in JSON format, and (ii) a [Game of Thrones data set](https://github.com/chrisalbon/war_of_the_five_kings_dataset) in CSV format.

**Example**: [example1/example1.ipynb](example1/example1.ipynb) 

## 2. Defining, Training, and Testing Models

### Regression

To make correct predictions, we need to understand the relationships between variables in our data and model this using statistical methods. One of those methods is called regression. In this example notebook, we are going to create a regression model to predict the weights of baseball players based on their height.

**Example**: [example2/example2.ipynb](example2/example2.ipynb) 

**Exercise** - Try using `github.com/sajari/regression` to train our regression model instead of `gonum`. *Hint* - look at the examples [here](https://github.com/sajari/regression). When you are ready, you can look at our solution notebook [here](solutions/solution1.ipynb). 

### Classification

Sometimes we don't need to predict a continuous value (like weight, stock price, or temperature). We might need to predict whether some observation belong to one or more discrete labels/classes (e.g., fraud or not fraud). In this example notebook, we will create a couple of classification models that predict flower species from physical measurements of flowers.

**Example**: [example3/example3.ipynb](example3/example3.ipynb). 

**Exercise** - Test out our kNN model for multiple *k* values to determine what *k* value we should use. When you are ready, you can look at our solution notebook [here](solutions/solution2.ipynb). 

### Clustering

Both regression and classification are considered "supervised" learning techniques, where we are trying to predict something based on labeled examples of that thing. However, there are also "unsupervised" learning techniques to, for example, detect groupings in your data set when you don't know what groups exist. This is called clustering, and we will look at one clustering algorithm in the following notebook called k-means.

**Example**: [example4/example4.ipynb](example4/example4.ipynb). 

**Exercise** - TBD


## 3. (Bonus) More sophisticated models

Sometimes we may need a model that is more complicated than linear regression or kNN. Go has us covered here. We can interface with major frameworks like TensorFlow, utilize more Go-centric frameworks like Gorgonia, or utilize services like MachineBox.  Moreover, in some cases (e.g., streaming ML analysis) we may want to leverage Go's built in concurrency primitives. 

[This bonus material](bonus), which we may or may not get to cover in the workshop, provides some more info about these methods and gives a couple of examples.

## Resources

- [Machine Learning with Go](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-go)  

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
