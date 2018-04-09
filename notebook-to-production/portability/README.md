# Making our pipeline stages portable

This material introduces you to one way of making each stage of your AI pipeline portable, Docker. Once you are done with this material you will know how to port local Python code to cloud/on-prem instances via Docker. 

This guide will walk you through:

1. [An overview of Docker](#1-overview-of-docker)
2. [Docker jargon](#2-docker-jargon)
3. [Docker-ize our pre-processing stage](#3-docker-ize-our-pre-processing-stage)
4. [Run a Docker container](#4-run-a-docker-container)

Bonus:

1. [Push your Docker image to a registry](README.md#1-push-your-docker-image-to-a-registry)
2. [Docker-ize and run our training stage](#2-docker-ize-and-run-our-training-stage)
3. [Docker-ize and run our inference stage](#3-docker-ize-and-run-our-inference-stage)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Overview of Docker  

Ok, we have our code for model training, inference, and pre-processing and we need to:

- scale this code up to larger data sets,
- run it automatically at certain times or based on certain events, 
- share it with teammates so they can generate their own results, or
- connect it to other code running in our company's infrastructure.

How can we do this with a high degree of reproducibility and operational/computation efficiency? And how can we ensure that our engineering team doesn't hate the data science team because they always have to deploy data science things in a "special" way with "special" data science tools. 

Docker solves many of these issues and even has additional benefits because it leverages *software containers* as it primary way of encapsulating applications. Containers existed before Docker, but Docker has made containers extremely easy to use and accesible. Thus many just associate software containers with Docker containers. When working with Docker containers, you might see some similarities to VMs, but they are quite different:

![Alt text](https://blog.netapp.com/wp-content/uploads/2016/03/Screen_Shot_2016-03-11_at_9.14.20_PM1.png)

As you can see Docker containers have the following unique properties which make them extremely useful:

- They don't include an entire guest OS. They just include your application and the associated libraries, file system, etc. This makes them much smaller than VMs (some of my Docker containers are just a few Mb). This also makes spinning up and tearing down containers extremely quick.
- They share an underlying host kernel and resources. You can spin up 10s or 100s or Docker containers on a single machine. They will all share the underlying resources, such that you can efficiently utilize all of the resources on a node (rather than statically carving our resource per process). 

This is why Docker containers have become so dominant in the infrastructure world. Data scientists and AI researchers are also latching on to these because they can:

- Docker-ize an application quickly, hand it off to an engineering organization, and have them run it in a manner similar to any other application.
- Experiment with a huge number of tools (Tensorflow, PyTorch, Spark, etc.) without having to install anything other than Docker.
- Manage a diverse set of data pipeline stages in a unified way.
- Leverage the huge number of excellent infrastructure projects for containers (e.g., those powering Google scale work) to create application that auto-scale, self-heal, are fault tolerant, etc.
- Easily define and reproduce environments for experimentation.

## 2. Docker Jargon

Docker jargon can sometimes be confusing, so let's go ahead and define some key terms. Refer back to this list later on the CodeLab if you need to:

- Docker *Image* - the bundle that includes your app & dependencies
- Docker *Container* - a running instance of a Docker image
- *Docker engine* - the application that builds and runs images
- Docker *registry* - where you store, tag, and get pre-built Docker images
- *Dockerfile* - a file that tells the engine how to build a Docker image

Thus, a common workflow when building a Docker-ized application is as follows:

1. Develop the application (as you normally would)
2. Build a Docker image for the app with Docker engine
3. Upload the image to a registry
4. Deploy a Docker container, based on the image from the registry, to a cloud instance or on premise node

## 3. Docker-ize our pre-processing stage

Let's say that we want to "Docker-ize" [example1/pre_process.py](example1/pre_process.py). This means that we want to create a *Docker image*. The Docker image will include our application and any library/package dependencies. Once built, we can then run this Docker image as a *Docker container* on any machine that is running the *Docker engine*. Regardless of the host OS or local configuration, this will buy us predictable behavior and portability.

To Docker-ize `pre_process.py` we need a *Dockerfile* that will tell Docker how to build the Docker image. Our example Dockerfile is included [here](example1/Dockerfile). Once we have that Dockerfile, we can build the Docker image by doing the following from the [example1](example1) directory:

```
$ sudo docker build -t pre-process . 
```

This command will likely ask you for a sudo password. This will be the same password given earlier in the workshop to connect to the instance.

Once built, you should be able to see your image in the list of images on the machine via:

```
$ sudo docker images
```

## 4. Run a Docker container

After building the Docker image, we can run this Docker image on any machine that has Docker installed. Let's try just running it on this instance with:

```
$ sudo docker run -it pre-process /bin/bash
```

This will open up an interactive bash shell in the container. You can explore the container a bit and try running the code (`pre_process.py`) that we added to the image.

*Note* - There are a bunch of ways to run Docker images. We will discuss some of these in class. However, for a more in depth intro check out [this codelab](https://github.com/dwhitena/qcon-ai-docker-workshop).

## Bonus exercises

You may not get to all of these bonus exercises during the workshop time, but you can perform these and all of the above steps any time you like with a [local installation of Docker](https://www.docker.com/community-edition). 

### 1. Push your docker image to a registry

Generally, Docker images are stored and versioned in a *Docker registry*. [Docker Hub](https://hub.docker.com/) is an example of such a registry and is pretty convenient for development. If you have a Docker Hub account or if you create one, we can push our previously built image to the registry by (i) tagging the image with our Docker Hub username, and (ii) logging into Docker Hub, and (iii) pushing the image to Docker Hub.

You should be able to replace `dwhitena` below with your Docker Hub username:

```
$ sudo docker tag pre-process dwhitena/pre-process
$ sudo docker login
Login with your Docker ID to push and pull images from Docker Hub. If you don't have a Docker ID, head over to https://hub.docker.com to create one.
Username: dwhitena
Password:
Login Succeeded
$ sudo docker push dwhitena/pre-process
The push refers to a repository [docker.io/dwhitena/pre-process]
d370dc3d40b4: Pushed
3e205a7e1532: Pushed
3523755f4e34: Pushed
9e17bfee4bf6: Pushed
cdcaace38a54: Pushed
6e1b48dc2ccc: Pushed
ff57bdb79ac8: Pushed
6e5e20cbf4a7: Pushed
86985c679800: Pushed
8fad67424c4e: Pushed
latest: digest: sha256:fd28b3993efcbaeb4ceb6885435839483d5cc74360565db16b9708df7ee741f7 size: 2427
```

### 2. Docker-ize and run our training stage 

Similar to the pre-processing stage, we include a Dockerfile for `train.py` in [bonus2](bonus2). Try building and running this Docker image. What's different about this Docker image as compared to the one for pre-processing.

### 3. Docker-ize and run our inference stage

Similar to the pre-processing stage, we include a Dockerfile for `infer.py` in [bonus3](bonus3). Try building and running this Docker image. 

## Resources

- [Getting started with Docker](https://docs.docker.com/get-started/)
- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)
- [Docker-izing your AI applications, CodeLab](https://github.com/dwhitena/qcon-ai-docker-workshop)
