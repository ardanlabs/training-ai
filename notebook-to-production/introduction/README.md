## Introduction to productionizing ML/AI

This material introduces some common pain points and pitfalls that people fall into when trying to productionize data science work. Once your are done with this material, you will understand what the common pain points are and the guiding principles that will help us overcome them.

Much of this section of the course is meant to be a discussion. That discussion will be centered around an example AI workflow in a Jupyter notebook. This guide will walk you through:

1. [Connecting to your workshop instance](#connecting-to-your-workshop-instance)


You should have been given an IP for a remote machine at the beginning of the course.  The remote machine already has Jupyter, scikit-learn, PyTorch, Docker, etc. installed along with all of the command line tools we will be needing throughout the course.  To log into the remote machine on Linux or Mac, open and terminal and:

```
$ ssh pachrat@<remote machine IP>
```

On Windows you can use PuTTY or another ssh client.  You will be asked for a password, which you should also be given during the workshop.  To verify that everything is running correctly on the machine, you should be able to open a Python terminal and run the following with the corresponding response:

```
$ docker --version
Docker version 1.11.2, build b9f10c9
```

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
