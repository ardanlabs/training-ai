# Introduction to Python tooling and ML/AI workflows

This material introduces some of the commonly used Python tooling for data science and ML/AI. It also introduces the ML/AI model development workflow. Once you are done with this material, you will understand what sets of tools are used in producing AI models, and how data scientists often interact with those tools.

You can follow this guide as we work through the material in class. Most of the commands/instructions that will be given in class are repeated here, so you can follow along and/or catch up when needed. Specifically, this guide will walk you through:

1. [Connecting to your workshop instance](#1-connecting-to-your-workshop-instance)
2. [Cloning the workshop material](#2-cloning-the-workshop-material)
3. [Starting Jupyter](#3-starting-jupyter)
4. [Dealing with data](#4-dealing-with-our-data)
5. [Developing a ML/AI model in Python](#5-developing-a-mlai-model-in-python)
6. [The model development/utilization workflow](#6-the-model-developmentutilization-workflow)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Connecting to your workshop instance

You should have been given an IP for a remote machine at the beginning of the course.  The remote machine already has Jupyter, scikit-learn, PyTorch, Docker, etc. installed along with all of the command line tools we will be needing throughout the course.  To log into the remote machine on Linux or Mac, open and terminal and:

```
$ ssh pachrat@<remote machine IP>
```

On Windows you can use PuTTY or another ssh client.  You will be asked for a password, which you should also be given during the workshop.  To verify that everything is running correctly on the machine, you should be able to open a Python terminal by running

```
$ python3
```

and then run the following with a similar response:

```
>>> import pandas as pd
>>> pd.DataFrame([[0,1],[1,0]], columns=['a', 'b'])
   a  b
0  0  1
1  1  0
>>>
```

## 2. Cloning the workshop material

Once, you are logged into your workshop instance, you will need to retrieve the workshop materials from this git repo. That way, we will all be working off of the same code templates and notebooks. To clone the repo, you can run:

```
$ git clone https://github.com/ardanlabs/training-ai.git
```

This will pull down the workshop materials to your instance. To confirm that the materials are there, you can navigate to the `training-ai/notebook-to-production` directory and the list contents. You should see:

```
$ cd training-ai/notebook-to-production/
$ ls
deploying_managing  frameworks_that_scale  introduction  pipeline_stages  portability  README.md
```

## 3. Starting Jupyter

Now, we are going to start our journey to production with a familiar Python tools. The first of those (which isn't necessarily specific to Python, but is Python focused) is [Jupyter](http://jupyter.org/). Navigate to the `introduction` directory and then start Jupyter as follows:

```
$ cd introduction
$ jupyter notebook --no-browser --port 8888 --ip=* --NotebookApp.token=''
```

You will now be able to visit `<your-instance-IP>:8888` in a browser to use Jupyter. When you are ready to stop using Jupyter, you can type `CTRL+c` in the terminal to stop Jupyter.

## 4. Dealing with our data

While you have Jupyter up and running, click on the `example1_data_munging.ipynb` notebook (which you should see in the `example1` directory). This will bring up our example notebook for parsing and manipulating data. If you are new to Jupyter notebooks, you can:

- click in any code block/cell to modify or run that code
- type `shift+enter` to execute a code block (or use the widgets at the top of the UI)
- if you get in a pickle, you might try selecting the "Kernel" menu at the top of the UI and then select "Restart Kernel" or similar.

We will run through and discuss this notebook interactively in class. 

**Exercise** - This brings us to our first official "exercise" in the course. Instead of looking at a pre-baked solution that implements the above steps, try it out on your own! To do this:

    - Start Jupyter again (if it's not still running),
    - Navigate to the [template1](exercises/template1) directory, and
    - Open up the `template1_` notebook.

This template notebook has some comments near the bottom where you need to fill in the missing pieces. Try to fill in these pieces without looking at the `solution1_` notebook under `exercises/solution1`, but don't feel bad if you can't get it. When you are ready, look at the solution1 notebook to see how I implemented these step (which is not the only or necessarily best solution). Once everyone has a chance to work on this, we will go over the solution together.

## 5. Developing a ML/AI model in Python

Our example problem for the day will be the [Iris flower classification problem](https://en.wikipedia.org/wiki/Iris_flower_data_set), and we will start by solving that problem using scikit-learn. Scikit-learn is the first ML/AI framework that many people use, and so we will use it as our jumping off point. Restart Jupyter (if you don't have it running), and open `example2_model_training.ipynb` from the `example2` directory. 

**Exercise** - Try implementing a model other than kNN with scikit-learn:

- [template](exercises/template2)
- [solution](exercises/solution2)

## 6. The model development/utilization workflow

Now that we have gotten our hands dirty building some models in Python, let's take a step back and think about the model development workflow in general. We will do this interactively in class via Q&A and via the class slides.

## Resources

Technical resources:

- [Jupyter](http://jupyter.org/)
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
