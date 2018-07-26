# Introduction and ML/AI workflows

This material introduces some of the infrastructure and tooling we will be using in the workshop. It also introduces the ML/AI model development workflow. Once you are done with this material, you will understand some of the tooling we will be using to develop Go ML/AI applications, and how data scientists develop models.

You can follow this guide as we work through the material in class. Most of the commands/instructions that will be given in class are repeated here, so you can follow along and/or catch up when needed. Specifically, this guide will walk you through:

1. [Connecting to your workshop instance](#1-connecting-to-your-workshop-instance)
2. [Cloning the workshop material](#2-cloning-the-workshop-material)
3. [Starting Jupyter](#3-starting-jupyter)
4. [The model development/utilization workflow](#6-the-model-developmentutilization-workflow)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Connecting to your workshop instance

You should have been given an IP for a remote machine at the beginning of the course.  The remote machine already has Jupyter, scikit-learn, PyTorch, Docker, etc. installed along with all of the command line tools we will be needing throughout the course.  To log into the remote machine on Linux or Mac, open and terminal and:

```
$ ssh pachrat@<remote machine IP>
```

On Windows you can use PuTTY or another ssh client.  You will be asked for a password, which you should also be given during the workshop.  To verify that everything is running correctly on the machine, you should be able to verify the Docker version installed on this instance by running

```
$ docker version
```

## 2. Cloning the workshop material

Once you are logged into your workshop instance, you will need to retrieve the workshop materials from this git repo. That way, we will all be working off of the same code templates, Dockerfiles, pipeline specifications, and notebooks. To clone the repo, you can run:

```
$ git clone https://github.com/ardanlabs/training-ai.git
```

This will pull down the workshop materials to your instance. To confirm that the materials are there, you can navigate to the `training-ai/machine-learning-with-go` directory and the list contents. You should see:

```
$ cd training-ai/machine-learning-with-go/
$ ls
README.md               building_an_ml_workflow ml_with_go              ml_workflow
```

## 3. Starting Jupyter

Now, we are going to start our journey in Go ML with a tool that is familiar to many ML/AI devs called [Jupyter](http://jupyter.org/). Navigate to the `introduction` directory and then start Jupyter as follows:

```
$ cd introduction
$ docker something...
```

You will now be able to visit `<your-instance-IP>:8888` in a browser to use Jupyter. When you are ready to stop using Jupyter, you can type `CTRL+c` in the terminal to stop Jupyter.

## 4. The model development/utilization workflow

Now that we have gotten our hands dirty building some models in Python, let's take a step back and think about the model development workflow in general. We will do this interactively in class via Q&A and via the class slides.

## Resources

Technical resources:

- [Jupyter](http://jupyter.org/)
- [gophernotes](https://github.com/gopherdata/gophernotes)

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
