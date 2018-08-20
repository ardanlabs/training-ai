# Building a complete Go-based ML workflow

This material walks you through a lab which will help you implement a full ML workflow with Go and Go-based tools, from data ingress to training to evaluation to inference. Once you are done with this material, you know how to implement the stages of the ML workflow in Go (for at least one type of ML model and data), and you will be able to transfer this workflow scaffolding to other problems.

![Alt text](https://docs.google.com/drawings/d/e/2PACX-1vQApl8ErSbCERlcPLBth0hxqYJwvIlKtLU4IrqtmnISCaTls4HRCsf6iKomscN5gdiWRLtAFVFqzlkC/pub?w=1165&h=662)

Specifically, this guide will walk you through:

1. [Introduction to the Problem/Data](#1-introduction-to-the-problem-and-data)
2. [Exploratory Model Development](#1-exploratory-model-development)
3. [Preparation of scalable data pipeline stages](#2-preparation-of-scalable-data-pipeline-stages)
4. [Deploying the data pipeline](#3-deploying-the-data-pipeline)
5. [Managing, updating, and scaling the pipeline](#6-managing-updating-and-scaling-the-pipeline)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Introduction to the problem and data

The problem that we will be working on in this lab is the prediction of disease progression using regression. The data that we will be leveraging is data about the progression of the disease diabetes in a number of patients:

```
age,sex,bmi,map,tc,ldl,hdl,tch,ltg,glu,y
0.0380759064334,0.0506801187398,0.0616962065187,0.021872354995,-0.0442234984244,-0.0348207628377,-0.043400845652,-0.00259226199818,0.0199084208763,-0.0176461251598,151.0
-0.00188201652779,-0.044641636507,-0.0514740612388,-0.0263278347174,-0.00844872411122,-0.0191633397482,0.0744115640788,-0.0394933828741,-0.0683297436244,-0.0922040496268,75.0
0.0852989062967,0.0506801187398,0.0444512133366,-0.00567061055493,-0.0455994512826,-0.0341944659141,-0.0323559322398,-0.00259226199818,0.00286377051894,-0.0259303389895,141.0
-0.0890629393523,-0.044641636507,-0.0115950145052,-0.0366564467986,0.0121905687618,0.0249905933641,-0.0360375700439,0.0343088588777,0.0226920225667,-0.00936191133014,206.0
0.00538306037425,-0.044641636507,-0.0363846922045,0.021872354995,0.00393485161259,0.0155961395104,0.00814208360519,-0.00259226199818,-0.0319914449414,-0.0466408735636,135.0
-0.0926954778033,-0.044641636507,-0.0406959405,-0.0194420933299,-0.0689906498721,-0.0792878444118,0.041276823842,-0.07639450375,-0.041180385188,-0.0963461565417,97.0
-0.04547247794,0.0506801187398,-0.0471628129433,-0.0159992226361,-0.0400956398498,-0.0248000120604,0.000778807997018,-0.0394933828741,-0.0629129499163,-0.038356659734,138.0
0.0635036755906,0.0506801187398,-0.00189470584028,0.0666296740135,0.0906198816793,0.108914381124,0.0228686348215,0.0177033544836,-0.0358167281015,0.00306440941437,63.0
0.0417084448844,0.0506801187398,0.0616962065187,-0.0400993174923,-0.013952535544,0.00620168565673,-0.0286742944357,-0.00259226199818,-0.0149564750249,0.011348623244,110.0
```

Each row in the data set includes a certain patient's `age`, `sex`, `bmi` (body mass index), and other health-related features along with an indication of diabetes disease progression (labeled as `y`). Our goal in this lab is to develop, deploy, manage, scale, and update a model that takes in one or more of these patient features and generates a prediction for diabetes progression.

To successfully complete the lab, we will need go through the steps of our model development workflow (introduced in the first section of the class), including:

- Exploratory data gathering, profiling, and model development
- Preparation of scalable data pipeline stages
- Deployment of the scalable data pipeline stages in a data pipeline
- Scaling, updating, and quality control of those pipeline stages

## 2. Exploratory Model Development

To get started, we need to profile our data and interactively figure out the parameters we want to use in our model. We will do this with a Jupyter notebook. Open Jupyter via Docker on your workshop instance (similar to earlier) and then navigate to the [exercise1/template/template1.ipynb](exercise1/template/template1.ipynb) notebook. 

This template notebook has some of the exploratory work done for you, and it points out areas where you can practice the remaining work (via code comments beginning with  `EXERCISE`). Try to implement these bits of work based on our previous example and based on the relevant godocs for the various packages. 

Don't worry if you can't get through everything. The point is to practice and get your hands dirty, not to finish all of the pieces. Feel free to take a look at the [exercise1/solution/solution1.ipynb](exercise1/solution/solution1.ipynb) notebook to see how we implemented the solutions to the exercises. These are meant to be examples of solutions, but they are by no means the only solutions or the best ones. 

## 3. Preparation of scalable data pipeline stages

### Pipeline stages

Jupyter gives us a really nice interface to interactively explore our data and figure out the type of modeling that we want to employ. However, a Jupyter notebook isn't really "deployable" within a company's infrastructure. If you want to actually get value out of your ML work, you need to deploy it off of your laptop, integrate it with other applications, make it scalable/manageable, and run it in an automated manner.

To this end, let's split up the pieces of our workflow, in our Jupyter notebook, into a series of Go programs that we will run as stages of a "data pipeline." Let's create the following:

1. *Training* - a program that takes in a specified training data set, trains our model, and outputs a JSON representation of our model,
2. *Quality control* - one or more programs that takes in the JSON representation of our model and quality controls it against a holdout data set (in our solution we create one program to pre-process our QC data and another to do the QC check on the model), and
3. *Inference* - a program that takes in the JSON representation of our model and utilizes the model to make predictions based on input JSON files that include patient attributes.

![Alt text](https://docs.google.com/drawings/d/e/2PACX-1vQT3oCUOAt85LkTyuv2onbty8OJMc4H_jBJCSxjEpPzIg9pFtKGhNJlK2xDEn-vZIpCg0hl_ZBjsDKT/pub?w=1654&h=517)

To get you started, we've created templates for these programs. We recommend that you pull down the data set under [data/diabetes.csv](data/diabetes.csv) (or at least a sample) and work off of these templates locally to develop your pipeline stages. Use your favorite IDE and `go get` any of the packages that you might not have locally. 

As with our previous notebook, we have also created solutions for each of the stages. Feel free to look at these while you develop, and, again, don't worry if you don't get through all of the examples. You can utilize our pre-baked solutions later on when we deploy the pipeline, so you won't get left behind. You can always revisit this repo later on to retry any of the exercises.

1. *Training* - [template](exercise2/templates/template1/template1.go), [solution](exercise2/solutions/solution1/solution1.go)
2. *Quality control* - [template](exercise2/templates/template2), [solution](exercise2/solutions/solution2)
3. *Inference* - [template](exercise2/templates/template3/template3.go), [solution](exercise2/solutions/solution3/solution3.go)

### Docker images

We also need to make our pipeline stages portable. That is, we need to be able to run them on any arbitrary infrastructure with reproducible behavior. Thankfully, we have a tool built with Go that solves that problem for us, *Docker*. If you haven't used Docker before, don't worry. We will discuss a bit in class and we have pre-baked Docker images that you will be able to use in the lab.

However, if you get done with the above exercise creating the pipeline stages, go ahead and try to Docker-ize these (i.e., create a Dockerfile and build a docker image for each respective stage). If you our curious, you can look at our example Dockerfiles and makefiles in the [solutions](exercise2/solutions) directory, and/or experiment with our pre-built docker images for each stage:

- *Training* - `gopherdata/gc2018:training`
- *Quality Control pre-processing* - `gopherdata/gc2018:qcpre`
- *Quality Control model check* - `gopherdata/gc2018:qcontrol`
- *Inference* - `gopherdata/gc2018:inference`

## Deploying the data pipeline

Ok, we now have multiple Docker images that will allow us to deploy our data pipeline stages on some infrastructure (e.g., in the cloud), but how are we going to orchestrate and schedule all of that work? Well, as you probably know, the Go community has developed the now de facto standard container scheduler called Kubernetes, and we will be using it here!

Kubernetes will allow us to deploy our data pipeline stages as containers, but it doesn't have the built in functionality we need to tie our stages together in a pipeline or to store/access/modify the data associated with our pipeline. For this, we will utilize yet another tool built with Go called Pachyderm. Pachyderm gives us that layer we need on top of Kubernetes for data pipelining and data management. 

We have already connected your workshop instances to a running Kubernetes cluster with Pachyderm, so ssh back into the workshop instance if you aren't there already. Then you can follow [this guide](exercise3/README.md) to deploy the following Pachyderm pipeline on top of Kubernetes:

![Alt text](https://docs.google.com/drawings/d/e/2PACX-1vSEuTeHVqRTzAmuQBILI_KExrF0oZsWJ_FaclWONTM60E_e5KfNUHAqK8S5L92R4AKjcmMWipkouUYd/pub?w=1537&h=612)

## Managing, updating, and scaling the pipeline

Congrats! You have deployed a full Go-based ML pipeline to a k8s cluster! Now let's try to update certain parts of this pipeline, scale particular stages, and monitor what happens. These are essential elements that need to be considered when putting ML in production.

Follow [this guide](exercise4/README.md) to see how this works in our k8s-based approach.

## Resources

Technical resources:

- [Pachyderm](http://pachyderm.io/)
- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/)

___
All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
