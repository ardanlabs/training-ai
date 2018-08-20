# Introduction and ML/AI workflows

This material introduces some of the infrastructure and tooling we will be using in the workshop. It also introduces the major types of machine learning along with a development workflow used for model development. Once you are done with this material, you will have a high level understanding of the landscape of machine learning techniques and the process that data scientists employ when developing models. 

You can follow this guide as we work through the material in class. Most of the commands/instructions that will be given in class are repeated here, so you can follow along and/or catch up when needed. Specifically, this guide will walk you through:

1. [Connecting to your workshop instance](#1-connecting-to-your-workshop-instance)
2. [Cloning the workshop material](#2-cloning-the-workshop-material)
3. [Starting Jupyter](#3-starting-jupyter)
4. [Intro to ML and the ML workflow](#6-intro-to-ml-and-the-ml-workflow)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Connecting to your workshop instance

You should have been given an IP for a remote machine at the beginning of the course.  The remote machine already has Jupyter, scikit-learn, PyTorch, Docker, etc. installed along with all of the command line tools we will be needing throughout the course.  To log into the remote machine on Linux or Mac, open and terminal and run:

```
$ ssh pachrat@<remote machine IP>
```

On Windows you can use PuTTY or another ssh client.  You will be asked for a password, which you should also be given during the workshop.  To verify that everything is running correctly on the machine, you should be able to verify the Docker version installed on this instance by running:

```
$ docker version
```

## 2. Cloning the workshop material

Once you are logged into your workshop instance, you will need to retrieve the workshop materials from this git repo. That way, we will all be working off of the same code templates, Dockerfiles, pipeline specifications, and notebooks. To clone the repo, you can run:

```
$ git clone https://github.com/ardanlabs/training-ai.git
```

This will pull down the workshop materials to your instance. To confirm that the materials are there, you can navigate to the `training-ai/machine-learning-with-go` directory and list the contents. You should see something like:

```
$ cd training-ai/machine-learning-with-go/
$ ls
README.md   ml_intro    ml_with_go  ml_workflow
```

## 3. Starting Jupyter

Now, we are going to start our journey in Go ML with a tool that is familiar to many ML/AI devs called [Jupyter](http://jupyter.org/). To make this easy, the organizers have create a docker image with Jupyter and a Go kernel for Jupyter called `gophernotes`. You could run Jupyter locally for this sort of development work (which would be the typical case), but for this workshop just run the following from your workshop instance:

```
$ docker run -it -p 8888:8888 -v /home/pachrat/training-ai/machine-learning-with-go:/notebooks dwhitena/gophernotes:gc
```

You will now be able to visit `<your-workshop-instance-IP>:8888` in a browser to use Jupyter. When you are ready to stop using Jupyter, you can type `CTRL+c` in the terminal to stop Jupyter.

## 4. Intro to ML and the ML workflow

Before we push on into some data munging and ML model development, let's take a step back and think about machine learning in general and the model development workflow. This will give us a solid baseline and will put the hands-on work we will do in the next section in context. 

We will do this interactively in class via Q&A and via the class slides.

## Resources

Technical resources:

- [Jupyter](http://jupyter.org/)
- [gophernotes](https://github.com/gopherdata/gophernotes)

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
