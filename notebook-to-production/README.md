# Python-based workflows - from notebook to production

This material is for intermediate-level data scientists, developers, data engineers, or researchers. Specifically, this material is for those who have some experience developing ML/AI models on sample data sets (maybe in Jupyter), but who might struggle to scale, deploy, and productionize their work. They need to understand which Python tools to use as they scale our workflows beyond the notebook, and they need to understand how to manage and distribute work on large data.

- [Slides from the class](https://docs.google.com/presentation/d/1vhINmKo-gIoDU1vVDqpg51auZpPkFV0h_X-ukdyYsFc/edit?usp=sharing)
- Instructor - Daniel Whitenack
  - [website/blog](http://www.datadan.io/)
  - [twitter](https://twitter.com/dwhitena)
  - [github](https://github.com/dwhitena)
- Prerequisties/getting started:
  - You will need to ssh into a cloud instance. Remind yourself of how to do that and install a client if needed:
    - On a Mac or Linux machine, you should be able to ssh from a terminal (see these [Mac instructions](http://accc.uic.edu/answer/how-do-i-use-ssh-and-sftp-mac-os-x) and [Linux instructions](https://www.digitalocean.com/community/tutorials/how-to-use-ssh-to-connect-to-a-remote-server-in-ubuntu)).
    - On a Windows machine, you can either [install and use an ssh client (I recommend PuTTY)](https://www.putty.org/) or [use the WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
  - You will also need to work a bit at the command line. If you are new to the command line or need a refresher, look through [this quick tutorial](https://lifehacker.com/5633909/who-needs-a-mouse-learn-to-use-the-command-line-for-almost-anything).
- If you need further help productionizing ML/AI workflows, want to bring this class to your company, or just have ML/AI related questions, [Ardan Labs](https://www.ardanlabs.com/) is here to help! Reach out to the instructor using the links above or via the [Ardan Labs website](https://www.ardanlabs.com/). 

*Note: This material has been designed to be taught in a classroom environment. The code is well commented but missing some of the contextual concepts and ideas that will be covered in class.*

## Intro to productionizing ML/AI 

This material introduces some common pain points and pitfalls that people fall into when trying to productionize data science work. Once your are done with this material, you will understand what the common pain points are and the guiding principles that will help us overcome them.    

[Introduction to productioning ML/AI](introduction)

## Using frameworks that scale

This material introduces some methods and frameworks that will help our workflow scale beyond local sample data. Once you are done with this material, you will be exposed to some of the more scalable Python frameworks in the ecosystem (e.g., PyTorch) and have some experience refactoring modeling code for production.

[Using frameworks that scale](frameworks_that_scale)

## Breaking our workflow up into pipeline stages

This material walks you through breaking up a workflow, contained in a Jupyter notebook, into separate, scalable pipeline stages. Once you are done with this material, you will understand which portions of a ML/AI pipeline might benefit from being managed in isolation. You will also get some experience writing code for specific stages of a data pipeline (pre-processing, training, inference).

[Breaking our workflow up into pipeline stages](pipeline_stages)

## Making our pipeline stages portable

This material introduces you to one way of making each stage of your AI pipeline portable, Docker. Once you are done with this material you will know how to port local Python code to cloud/on-prem instances via Docker. 

[Making our pipeline stages portable](portability)

## Deploying, scaling, and managing our pipeline

This material introduces you to methods for orchestrating a multi-stage AI pipeline at scale. Once you are done with this material, you will understand various methods for deploying multi-stage pipelines along with their trade offs. You will also get hands-on experience deploying a multi-stage AI pipeline on a remote cluster.

[Deploying, scaling, and managing our pipeline](deploying_managing)

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
