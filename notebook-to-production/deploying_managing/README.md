# Deploying, scaling, and managing our pipeline

This material introduces you to methods for orchestrating a multi-stage AI pipeline at scale. Once you are done with this material, you will understand various methods for deploying multi-stage pipelines along with their trade offs. You will also get hands-on experience deploying a multi-stage AI pipeline on a remote cluster.

This guide will walk you through:

1. [Discussing AI pipeline orchestration, intro to k8s, KubeFlow, Seldon, and Pachyderm](#1-discussing-ai-pipeline-orchestration)
2. [Connecting to your Pachyderm cluster](#2-connect-to-your-pachyderm-cluster)
3. [Creating the input "data repositories"](README.md#3-create-the-input-data-repositories)
4. [Committing the training data set into Pachyderm](README.md#4-commit-the-training-data-set-into-pachyderm)
5. [Creating the pre-processing pipeline](#5-create-the-pre-processing-pipeline)
6. [Creating the training pipeline](README.md#6-create-the-training-pipeline)
7. [Committing input attributes](README.md#7-commit-input-attributes)
8. [Creating the inference pipeline](README.md#8-create-the-inference-pipeline)
9. [Examining the results](README.md#9-examine-the-results)

Bonus:

10. [Parallelizing the inference](README.md#10-parallelize-the-inference)
11. [Updating the model or training data set](README.md#11-update-the-model-or-training-data-set)
12. [Examining pipeline provenance](README.md#12-examine-pipeline-provenance)

It also includes a [list of resources](#resources) for those that want to dive in a little bit deeper.

## 1. Discussing AI pipeline orchestration

We now have our individual pipeline stages split up, Docker-ized, and portable. But how the heck do we manage them and deploy them scale? To answer that we need to have a little discussion. Let's try to figure out how we are going to:

- Get the right Docker images running on the right resources
- Get the right data to the right Docker images at the right time
- Collect results
- Trigger pipeline stages at the right time
- Individually scale pipeline stages

At the end of this discussion, we will have some new tools in our tool belt for AI pipeline orchestration.

## 2. Connect to your Pachyderm cluster  

Your workshop instance should be connected to a running Pachyderm cluster. To check this, you should be able to run the following with the corresponding response:

```
$ pachctl version
COMPONENT           VERSION
pachctl             1.7.0
pachd               1.7.0
```

We will be working from this directory, so you can go ahead and navigate there:

```
$ cd ~/training-ai/notebook-to-production/deploying_managing
```

## 3. Create the input data repositories 

On the Pachyderm cluster running in your remote machine, we will need to create the two input data repositories (for our training data and input attributes).  To do this run:

```
$ pachctl create-repo training
$ pachctl create-repo attributes
```

As a sanity check, we can list out the current repos, and you should see the two repos you just created:

```
$ pachctl list-repo
NAME                CREATED             SIZE
attributes          4 seconds ago       0B
training            8 seconds ago       0B
```

## 4. Commit the training data set into pachyderm

We have our training data repository, but we haven't put our training data set into this repository yet. To get this data into Pachyderm, we run:

```
$ pachctl put-file training master iris.csv -f data/iris.csv
```

Then, you should be able to see the following:

```
$ pachctl list-repo
NAME                CREATED              SIZE
training            About a minute ago   4.308KiB
attributes          About a minute ago   0B
$ pachctl list-file training master
NAME                TYPE                SIZE
iris.csv            file                4.308KiB
```

## 5. Create the pre-processing pipeline

Next, we can create the `pre_process` pipeline stage to pre-process the data in the training repository. To do this, we just need to provide Pachyderm with [a JSON pipeline specification](pre-process.json) that tells Pachyderm how to process the data.  

Once you have `pre-process.json`, creating our `pre_process` pipeline is as easy as:

```
$ pachctl create-pipeline -f pre-process.json
```

Once the pipeline worker spins up (which you can watch via `kubectl get pods`), Pachyderm will automatically kick off a job to perform the pre-processing:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                STARTED            DURATION RESTART PROGRESS  DL UL STATE
da28d87632e44898be301d6baf13a5ee pre_process/cd94a0be5558492d9dbf2e5d973c8b0a About a minute ago -        0       0 + 0 / 1 0B 0B running
```

That job should complete pretty quickly:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                STARTED            DURATION       RESTART PROGRESS  DL       UL       STATE
da28d87632e44898be301d6baf13a5ee pre_process/cd94a0be5558492d9dbf2e5d973c8b0a About a minute ago About a minute 0       1 + 0 / 1 4.308KiB 10.82KiB success
```

Once it is complete, you will notice that you have a new data repository for the output from this stage of our pipeline, and that data repository has our pre-processed data:

```
$ pachctl list-file pre_process master
NAME                TYPE                SIZE
x_train.csv         file                9.968KiB
y_train.csv         file                870B
```

## 6. Create the training pipeline

Similarly we can use [this JSON pipeline specification](train.json) to create our training pipeline:

```
$ pachctl create-pipeline -f train.json
```

Again, this will automatically kick off a training job (after the pod initializes and pull the Docker image, which could take a few minutes). Pachyderm will also gather the output (in this case, the serialized model):

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                STARTED       DURATION       RESTART PROGRESS  DL       UL       STATE
6ad8bc882c4748d5a308cab6e32866f2 train/df90f50ea4814db8866004163bdcd126       5 minutes ago -              0       0 + 0 / 1 0B       0B       running
da28d87632e44898be301d6baf13a5ee pre_process/cd94a0be5558492d9dbf2e5d973c8b0a 9 minutes ago About a minute 0       1 + 0 / 1 4.308KiB 10.82KiB success
$ pachctl list-job
ID                               OUTPUT COMMIT                                STARTED        DURATION       RESTART PROGRESS  DL       UL       STATE
6ad8bc882c4748d5a308cab6e32866f2 train/df90f50ea4814db8866004163bdcd126       5 minutes ago  5 minutes      0       1 + 0 / 1 10.82KiB 787B     success
da28d87632e44898be301d6baf13a5ee pre_process/cd94a0be5558492d9dbf2e5d973c8b0a 10 minutes ago About a minute 0       1 + 0 / 1 4.308KiB 10.82KiB success
$ pachctl list-repo
NAME                CREATED             SIZE
train               6 minutes ago       787B
pre_process         10 minutes ago      10.82KiB
training            14 minutes ago      4.308KiB
attributes          14 minutes ago      0B
$ pachctl list-file train master
NAME                TYPE                SIZE
model.pt            file                787B
```

## 7. Commit input attributes

Great! We now have a trained model that will infer the species of Iris flowers.  Let's commit some example flower attributes into Pachyderm that we would like to run through the model.  We have a couple examples under [data](data).  Feel free to use these or create your own.  To commit our samples, you can run:

```
$ pachctl put-file attributes master test1.csv -f data/test1.csv
$ pachctl put-file attributes master test2.csv -f data/test2.csv
```

You should then see:

```
$ pachctl list-file attributes master
NAME                TYPE                SIZE
test1.csv           file                96B
test2.csv           file                64B
```

## 8. Create the inference pipeline

We have another JSON blob, [infer.json](infer.json), that will tell Pachyderm how to perform the processing for the inference stage.  This is similar to our other JSON specifications except, in this case, we have two input repositories (the `attributes` and the `train`).  To create the inference stage, we simply run:

```
$ pachctl create-pipeline -f infer.json
```

This will kick off an inference job, because we have committed unprocessed attributes into the `attributes` repo.  The results will then be versioned in a corresponding `inference` data repository:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                STARTED        DURATION       RESTART PROGRESS  DL       UL       STATE
21306064197849008785bc80808421b7 inference/a9075053dba0442b954af6c4636894d8   18 seconds ago 14 seconds     0       2 + 0 / 2 1.693KiB 139B     success
6ad8bc882c4748d5a308cab6e32866f2 train/df90f50ea4814db8866004163bdcd126       10 minutes ago 5 minutes      0       1 + 0 / 1 10.82KiB 787B     success
da28d87632e44898be301d6baf13a5ee pre_process/cd94a0be5558492d9dbf2e5d973c8b0a 14 minutes ago About a minute 0       1 + 0 / 1 4.308KiB 10.82KiB success
$ pachctl list-repo
NAME                CREATED             SIZE
inference           47 seconds ago      139B
attributes          19 minutes ago      160B
pre_process         15 minutes ago      10.82KiB
train               11 minutes ago      787B
training            19 minutes ago      4.308KiB
```

## 9. Examine the results

We have created results from the inference, but how do we examine those results?  There are multiple ways, but an easy way is to just "get" the specific files out of Pachyderm's data versioning:

```
$ $ pachctl list-file inference master
NAME                TYPE                SIZE
test1.csv           file                85B
test2.csv           file                54B
$ pachctl get-file inference master test1.csv
Iris-versicolor
Iris-virginica
Iris-virginica
Iris-virginica
Iris-setosa
Iris-setosa
```

## Bonus exercises

You may not get to all of these bonus exercises during the workshop time, but you can perform these and all of the above steps any time you like with a [simple local Pachyderm install](http://docs.pachyderm.io/en/latest/getting_started/local_installation.html).  You can spin up this local version of Pachyderm is just a few commands and experiment with this, [other Pachyderm examples](http://docs.pachyderm.io/en/latest/examples/readme.html), and/or your own pipelines.

### 10. Parallelize the inference

You may have noticed that our pipeline specs included a `parallelism_spec` field.  This tells Pachyderm how to parallelize a particular pipeline stage.  Let's say that in production we start receiving a large number of flower attributes, and we need to keep up with our inference.  In particular, let's say we want to spin up 3 inference workers to perform inference in parallel.

This actually doesn't require any change to our code.  We can simply change our `parallelism_spec` to:

```
  "parallelism_spec": {
    "constant": "3"
  },
```

Pachyderm will then spin up 3 inference workers, each running our same `infer.py` script, to perform inference in parallel.  This can be confirmed by updating our pipeline and then examining the cluster:

```
$ vim infer.json 
$ pachctl update-pipeline -f infer.json
$ kubectl get pods
NAME                            READY     STATUS            RESTARTS   AGE
dash-67586ccc67-4ms9s           2/2       Running           0          18h
etcd-7dbb489f44-s9tfz           1/1       Running           0          18h
pachd-688c7bbbc6-w5kwh          1/1       Running           0          18h
pipeline-inference-v2-4bzrj     0/2       PodInitializing   0          3s
pipeline-inference-v2-576r2     0/2       Init:0/1          0          3s
pipeline-inference-v2-5khqk     0/2       Init:0/1          0          3s
pipeline-pre-process-v1-vfv2n   2/2       Running           0          18m
pipeline-train-v1-29k4j         2/2       Running           0          14m
$ kubectl get pods
NAME                            READY     STATUS    RESTARTS   AGE
dash-67586ccc67-4ms9s           2/2       Running   0          18h
etcd-7dbb489f44-s9tfz           1/1       Running   0          18h
pachd-688c7bbbc6-w5kwh          1/1       Running   0          18h
pipeline-inference-v2-4bzrj     2/2       Running   0          30s
pipeline-inference-v2-576r2     2/2       Running   0          30s
pipeline-inference-v2-5khqk     2/2       Running   0          30s
pipeline-pre-process-v1-vfv2n   2/2       Running   0          18m
pipeline-train-v1-29k4j         2/2       Running   0          14m
```

### 11. Update the model or training data set

Let's say that one or more observations in our training data set were corrupt or unwanted.  Thus, we want to update our training data set.  To simulate this, go ahead and open up `iris.csv` (e.g., with `vim`) and remove a couple of the reviews (i.e., the non-header rows).  Then, let's replace our training set:

```
$ vim data/iris.csv
$ pachctl put-file training master iris.csv -c -o -f data/iris.csv
```

Immediately, Pachyderm "knows" that the data has been updated, and it starts a new job to update the model and inferences:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                STARTED        DURATION       RESTART PROGRESS  DL       UL       STATE
682e9e8ff7ce4e8cb5150ac8518d17cf inference/79035d06cea24b4bbff76503a0019cc8   5 seconds ago  -              0       0 + 0 / 0 0B       0B       starting
1cd3967dfce64127b8d1a3313082a604 train/f28fd55c80364260af152a10258b406d       5 seconds ago  -              0       0 + 0 / 0 0B       0B       starting
2da4cda4cbf44f1da0e5152d824ad64d pre_process/7d05fa2c3c704b52b8d3fa24372e9813 5 seconds ago  -              0       0 + 0 / 1 0B       0B       running
e9ca0c7766754ed1b72a986233cf5127 inference/9e223462b5844a3a886b3aebf7566fea   2 minutes ago  7 seconds      0       0 + 2 / 2 0B       0B       success
21306064197849008785bc80808421b7 inference/a9075053dba0442b954af6c4636894d8   6 minutes ago  14 seconds     0       2 + 0 / 2 1.693KiB 139B     success
6ad8bc882c4748d5a308cab6e32866f2 train/df90f50ea4814db8866004163bdcd126       16 minutes ago 5 minutes      0       1 + 0 / 1 10.82KiB 787B     success
da28d87632e44898be301d6baf13a5ee pre_process/cd94a0be5558492d9dbf2e5d973c8b0a 20 minutes ago About a minute 0       1 + 0 / 1 4.308KiB 10.82KiB success
```

### 12. Examine pipeline provenance

Let's say that we have updated our model and/or training set a bunch of times.  Now we have multiple inferences that were made with different models and/or training data sets.  How can we know which results came from which specific models and/or training data sets?  This is called "provenance," and Pachyderm gives it to you out of the box.  

Suppose we have run the following jobs:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                STARTED            DURATION       RESTART PROGRESS  DL       UL       STATE
682e9e8ff7ce4e8cb5150ac8518d17cf inference/79035d06cea24b4bbff76503a0019cc8   About a minute ago 37 seconds     0       2 + 0 / 2 1.693KiB 139B     success
1cd3967dfce64127b8d1a3313082a604 train/f28fd55c80364260af152a10258b406d       About a minute ago 32 seconds     0       1 + 0 / 1 10.52KiB 787B     success
2da4cda4cbf44f1da0e5152d824ad64d pre_process/7d05fa2c3c704b52b8d3fa24372e9813 About a minute ago 4 seconds      0       1 + 0 / 1 4.198KiB 10.52KiB success
e9ca0c7766754ed1b72a986233cf5127 inference/9e223462b5844a3a886b3aebf7566fea   3 minutes ago      7 seconds      0       0 + 2 / 2 0B       0B       success
21306064197849008785bc80808421b7 inference/a9075053dba0442b954af6c4636894d8   7 minutes ago      14 seconds     0       2 + 0 / 2 1.693KiB 139B     success
6ad8bc882c4748d5a308cab6e32866f2 train/df90f50ea4814db8866004163bdcd126       17 minutes ago     5 minutes      0       1 + 0 / 1 10.82KiB 787B     success
da28d87632e44898be301d6baf13a5ee pre_process/cd94a0be5558492d9dbf2e5d973c8b0a 21 minutes ago     About a minute 0       1 + 0 / 1 4.308KiB 10.82KiB success
```

If we want to know which model and training data set was used for the latest inference, commit id `a9075053dba0442b954af6c4636894d8`, we just need to inspect the particular commit:

```
$ pachctl inspect-commit inference a9075053dba0442b954af6c4636894d8
Commit: inference/a9075053dba0442b954af6c4636894d8
Started: 8 minutes ago
Finished: 7 minutes ago
Size: 139B
Provenance:  attributes/4065496d3e144ccbadc7a27cbbe11033  pre_process/cd94a0be5558492d9dbf2e5d973c8b0a  train/df90f50ea4814db8866004163bdcd126  training/a9de1b71ead94c45aae64ec4e0e78226  __spec__/d514f9593f7241049b912ffb0edcdfa2  __spec__/2fd75d6eac6b4a7cbb8978cd39036340  __spec__/d177c5a636c844859e7bf4602026e70f
```

The `Provenance` tells us exactly which model and training set was used (along with which commit to reviews triggered the sentiment analysis).  For example, if we wanted to see the exact model used, we would just need to reference commit `960d0e24a836448fae076cf6f2c98e40` to the `train` repo:

```
$ pachctl list-file train df90f50ea4814db8866004163bdcd126
NAME                TYPE                SIZE
model.pt            file                787B
```

We could get this model to examine it, rerun it, revert to a different model, etc.

## Resources

KubeFlow:

- KubeFlow [org on GitHub](https://github.com/kubeflow/kubeflow)
- KubeFlow + Pachyderm + Seldon model CI/CD and inference pipeline example [here](https://github.com/dwhitena/examples/tree/master/github_issue_summarization/pachyderm_seldon_kvc)

Pachyderm:

- Join the [Pachyderm Slack team](http://slack.pachyderm.io/) to ask questions, get help, and talk about production deploys.
- Follow [Pachyderm on Twitter](https://twitter.com/pachydermIO), 
- Find [Pachyderm on GitHub](https://github.com/pachyderm/pachyderm), and
- [Spin up Pachyderm](http://docs.pachyderm.io/en/latest/getting_started/getting_started.html) in just a few commands to try this and [other examples](http://docs.pachyderm.io/en/latest/examples/readme.html) locally.

Kubernetes:

- [What is Kubernetes](https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/)
- [Tensorflow + GPUs on k8s](https://youtu.be/OZSA5hmkb0o)
