# Introduction to productionizing ML/AI

This material introduces some common pain points and pitfalls that people fall into when trying to productionize data science work. Once your are done with this material, you will understand what the common pain points are and the guiding principles that will help us overcome them.

Much of this section of the course is meant to be a discussion. That discussion will be centered around an example AI workflow in a Jupyter notebook. This guide will walk you through:

1. [Connecting to your workshop instance](#1-connecting-to-your-workshop-instance)
2. [Cloning the workshop material](#2-cloning-the-workshop-material)
3. [Starting Jupyter](#3-starting-jupyter)
4. [Running the example python workflow](#4-running-the-example-python-workflow)
5. [Discussing how we might productionize the workflow](#5-discussing-how-we-might-productionize-the-workflow)


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

This will pull down the workshop materials to your instance. To confirm that the materials are there, you can navigate to the training-ai directory and the list contents. You should see:

```
$ cd training-ai
$ ls
README.md               deploying_stages        introduction            pipeline_stages         scaling_managing_stages
```

## 3. Starting Jupyter

Now, we are going to start our journey to production with a familiar Python workflow contained in a jupyter notebook. Navigate to the `introduction` directory and then start Jupyter as follows:

```
$ cd introduction
$ jupyter notebook --no-browser --port 8888 --ip=* --NotebookApp.token=''
```

You will now be able to visit `<your-instance-IP>:8888` in a browser to use Jupyter. When you are ready to stop using Jupyter, you can type `CTRL+c` in the terminal to stop Jupyter.

## 4. Running the example Python workflow

While you have Jupyter up and running, click on the `example_sklearn_workflow.ipynb` notebook (which you should see in the `introduction` directory). This will bring up our example notebook. If you are new to Jupyter notebooks, you can:

- click in any code block/cell to modify or run that code
- type `shift+enter` to execute a code block (or use the widgets at the top of the UI)
- if you get in a pickle, you might try selecting the "Kernel" menu at the top of the UI and then select "Restart Kernel" or similar.

We will run through and discuss this notebook interactively in class. Our example problem for the day will be the Iris flower classification problem, and we will start by solving that problem using scikit-learn.

## 5. Discussing how we might productionize the workflow

It might not be clear what "productionize" means or how we might try doing that. This will be discussed in class, but before or during our conversation think about:

- What characteristics should a production AI workflow exhibit?
- What differentiates a production AI workflow from a non-production workflow?
- In what environments do production AI workflows run? 
- Are scaling and deployment the same thing?
- How might data change as we move to production?
- How do we want to manage our production AI workflows?
- Do we treat all parts of our workflow the same as we move them to production?
- What happens when our workflows fail at scale?
- How does data ingress/egress happen at scale?
- How will our workflow interact with other pieces of infrastruction or other applications?
- How do we handle our dependencies?

By the end of this discussion, we should decide on set of guidelines that should drive how we productionize our Python workflow.

## Resources

Blah….

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
