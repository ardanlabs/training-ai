# Deploying, scaling, and managing our pipeline

This material introduces you to methods for orchestrating a multi-stage AI pipeline at scale. Once you are done with this material, you will understand various methods for deploying multi-stage pipelines along with their trade offs. You will also get hands-on experience deploying a multi-stage AI pipeline on a remote cluster.

This guide will walk you through:

1. [Discussing AI pipeline orchestration, intro to k8s and Pachyderm](#1-discussing-ai-pipeline-orchestration)
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
11. [Updating the model training](README.md#11-update-the-model-training)
12. [Updating the training data set](README.md#12-update-the-training-data-set)
13. [Examining pipeline provenance](README.md#13-examine-pipeline-provenance)

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

You should have been given an IP for a remote machine at the beginning of the workshop.  The remote machine already has a Pachyderm cluster running locally and all of the command line tools we will be needing throughout the workshop.  To log into the remote machine, open and terminal and:

```
$ ssh pachrat@<remote machine IP>
```

You will be asked for a password, which you should also be given during the workshop.  To verify that everything is running correctly on the machine, you should be able to run the following with the corresponding response:

```
$ pachctl version
COMPONENT           VERSION             
pachctl             1.6.6           
pachd               1.6.6
```

We will be working from the `full_ml_workflows` directory, so you can go ahead and navigate there:

```
$ cd ~/amld-reproducible-ml-workshop/full_ml_workflows/
```

## 4. Create the input data repositories 

On the Pachyderm cluster running in your remote machine, we will need to create the two input data repositories (for our training data and input movie reviews).  To do this run:

```
$ pachctl create-repo training
$ pachctl create-repo reviews
```

As a sanity check, we can list out the current repos, and you should see the two repos you just created:

```
$ pachctl list-repo
NAME                CREATED             SIZE                
reviews             2 seconds ago       0 B                 
training            8 seconds ago       0 B
```

## 5. Commit the training data set into pachyderm

We have our training data repository, but we haven't put our training data set into this repository yet.  You can get the training data set that we will be using via:

```
$ wget https://s3.amazonaws.com/neon-workshop-data/labeledTrainData.tsv
```

This `labeledTrainData.tsv` file include 750 labeled movie reviews sampled from a larger IMDB data set.  Here we are using a sample for illustrative purposes (so our examples run a little faster in the workshop), but the entire data set can be obtained [here](https://www.kaggle.com/c/word2vec-nlp-tutorial/data).

To get this data into Pachyderm, we run:

```
$ pachctl put-file training master -c -f labeledTrainData.tsv
```

Then, you should be able to see the following:

```
$ pachctl list-repo
NAME                CREATED             SIZE                
training            6 minutes ago       977 KiB             
reviews             6 minutes ago       0 B                 
$ pachctl list-file training master
NAME                   TYPE                SIZE                
labeledTrainData.tsv   file                977 KiB
```

## 6. Create the training pipeline

Next, we can create the `model` pipeline stage to process the data in the training repository. To do this, we just need to provide Pachyderm with [a JSON pipeline specification](train.json) that tells Pachyderm how to process the data.  You can copy `train.json` from this repository or clone the whole repository to your remote machine.  After you have `infer.json`, creating our `model` training pipeline is as easy as:

```
$ pachctl create-pipeline -f train.json
```

Once the pipeline worker spins up, you will notice that Pachyderm has automatically kicked off a job to perform the model training:

```
$ pachctl list-job
ID                                   OUTPUT COMMIT STARTED        DURATION RESTART PROGRESS  DL UL STATE
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/-       11 seconds ago -        0       0 + 0 / 1 0B 0B running
```

This job should run for about 3-8 minutes.  In the mean time, we will address any questions that might have come up, help any users with issues they are experiences, and talk a bit more about ML workflows in Pachyderm.

After your model has successfully been trained, you should see:

```
$ pachctl list-job
ID                                   OUTPUT COMMIT                          STARTED        DURATION   RESTART PROGRESS  DL       UL       STATE
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/7d5eae8ba55c4f04b5a97e117eb93310 25 seconds ago 14 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
$ pachctl list-repo
NAME                CREATED             SIZE                
reviews             21 minutes ago      0 B            
model               10 minutes ago      20.93 MiB           
training            21 minutes ago      977 KiB             
$ pachctl list-file model master
NAME                TYPE                SIZE                
imdb.p              file                20.54 MiB           
imdb.vocab          file                393.2 KiB
```

## 7. Commit input reviews

Great! We now have a trained model that will infer the sentiment of movie reviews.  Let's commit some movie reviews into Pachyderm that we would like to run through the sentiment analysis.  We have a couple examples under [test](test).  Feel free to use these, find your own, or even write your own review.  To commit our samples (assuming you have cloned this repo on the remote machine), you can run:

```
$ cd test
$ pachctl put-file reviews master -c -r -f .
$ cd ..
```

You should then see:

```
$ pachctl list-file reviews master
NAME                TYPE                SIZE                
1.txt               file                770 B               
2.txt               file                897 B
```

## 8. Create the inference pipeline

We have another JSON blob, [infer.json](infer.json), that will tell Pachyderm how to perform the processing for the inference stage.  This is similar to our last JSON specification except, in this case, we have two input repositories (the `reviews` and the `model`) and we are using a different Docker image that contains `auto_inference.py`.  To create the inference stage, we simply run:

```
$ pachctl create-pipeline -f infer.json
```

This will immediately kick off an inference job, because we have committed unprocessed reviews into the `reviews` repo.  The results will then be versioned in a corresponding `inference` data repository:

```
$ pachctl list-job
ID                                   OUTPUT COMMIT                          STARTED                DURATION   RESTART PROGRESS  DL       UL       STATE
b5e9e4ba-93ef-4cfa-841b-cff3cbf21529 inference/-                            Less than a second ago -          0       0 + 0 / 2 0B       0B       running
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/7d5eae8ba55c4f04b5a97e117eb93310 4 minutes ago          14 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
$ pachctl list-job
ID                                   OUTPUT COMMIT                              STARTED        DURATION   RESTART PROGRESS  DL       UL       STATE
b5e9e4ba-93ef-4cfa-841b-cff3cbf21529 inference/fe985f6564da4edb9dd04892a911e7af 44 seconds ago 2 seconds  0       2 + 0 / 2 41.85MiB 70B      success
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/7d5eae8ba55c4f04b5a97e117eb93310     5 minutes ago  14 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
$ pachctl list-repo
NAME                CREATED              SIZE                
inference           About a minute ago   70 B                
reviews             2 hours ago          1.628 KiB           
model               About an hour ago    20.93 MiB           
training            2 hours ago          977 KiB
```

## 9. Examine the results

We have created results from the inference, but how do we examine those results?  There are multiple ways, but an easy way is to just "get" the specific files out of Pachyderm's data versioning:

```
$ pachctl list-file inference master
NAME                TYPE                SIZE                
1.txt               file                35 B                
2.txt               file                35 B                
$ pachctl get-file inference master 1.txt
Pred - [[ 0.50981182  0.49018815]]
$ pachctl get-file inference master 2.txt
Pred - [[ 0.53010267  0.46989727]]
```

Here we can see that each result file contains two probabilities corresponding to postive and negative sentiment, respectively.

## Bonus exercises

You may not get to all of these bonus exercises during the workshop time, but you can perform these and all of the above steps any time you like with a [simple local Pachyderm install](http://docs.pachyderm.io/en/latest/getting_started/local_installation.html).  You can spin up this local version of Pachyderm is just a few commands and experiment with this, [other Pachyderm examples](http://docs.pachyderm.io/en/latest/examples/readme.html), and/or your own pipelines.

### 10. Parallelize the inference

You may have noticed that our pipeline specs included a `parallelism_spec` field.  This tells Pachyderm how to parallelize a particular pipeline stage.  Let's say that in production we start receiving a huge number of movie reviews, and we need to keep up with our sentiment analysis.  In particular, let's say we want to spin up 10 inference workers to perform sentiment analysis in parallel.

This actually doesn't require any change to our code.  We can simply change our `parallelism_spec` to:

```
  "parallelism_spec": {
    "constant": "10"
  },
```

Pachyderm will then spin up 10 inference workers, each running our same `auto_inference.py` script, to perform inference in parallel.  This can be confirmed by updating our pipeline and then examining the cluster:

```
$ vim infer.json 
$ pachctl update-pipeline -f infer.json 
  AGE
dash-64bf867d6b-tmql2         2/2       Running           0          35m
etcd-7dbb489f44-dhjrf         1/1       Running           0          35m
pachd-9f4654f69-7fqzz         1/1       Running           0          35m
pipeline-inference-v2-289h8   2/2       Running           0          16s
pipeline-inference-v2-8t6dt   0/2       PodInitializing   0          16s
pipeline-inference-v2-d9f7p   0/2       Init:0/1          0          16s
pipeline-inference-v2-dsw4h   0/2       Init:0/1          0          16s
pipeline-inference-v2-gpbt8   0/2       Init:0/1          0          16s
pipeline-inference-v2-gwbq2   2/2       Running           0          16s
pipeline-inference-v2-htd44   0/2       Init:0/1          0          16s
pipeline-inference-v2-jg42j   0/2       PodInitializing   0          16s
pipeline-inference-v2-pshs9   0/2       PodInitializing   0          16s
pipeline-inference-v2-xntsr   0/2       Init:0/1          0          16s
pipeline-model-v1-src4q       2/2       Running           0          8m
$ kubectl get pods
NAME                          READY     STATUS    RESTARTS   AGE
dash-64bf867d6b-tmql2         2/2       Running   0          36m
etcd-7dbb489f44-dhjrf         1/1       Running   0          36m
pachd-9f4654f69-7fqzz         1/1       Running   0          36m
pipeline-inference-v2-289h8   2/2       Running   0          30s
pipeline-inference-v2-8t6dt   2/2       Running   0          30s
pipeline-inference-v2-d9f7p   2/2       Running   0          30s
pipeline-inference-v2-dsw4h   2/2       Running   0          30s
pipeline-inference-v2-gpbt8   2/2       Running   0          30s
pipeline-inference-v2-gwbq2   2/2       Running   0          30s
pipeline-inference-v2-htd44   2/2       Running   0          30s
pipeline-inference-v2-jg42j   2/2       Running   0          30s
pipeline-inference-v2-pshs9   2/2       Running   0          30s
pipeline-inference-v2-xntsr   2/2       Running   0          30s
pipeline-model-v1-src4q       2/2       Running   0          8m
```

### 11. Update the model training

**Note** - This exercise increases the training time of our model, so you might be waiting for 5+ minutes for the model to re-train (maybe up to 10-15 minutes).  If you don't want to wait for this amount of time during the workshop, you could try step 12, which will take less time.

You might have noticed that we only used one "epoch" in our model training the first time around.  This is probably not enough in general.  As such, you can change this to two, for example, by modifying `train.json`:

```
    "cmd": [ 
	"python", 
	"examples/imdb/train.py", 
	"-f", 
	"/pfs/training/labeledTrainData.tsv", 
	"-e", 
	"2", 
	"-eval", 
	"1", 
	"-s", 
	"/pfs/out/imdb.p", 
	"--vocab_file", 
	"/pfs/out/imdb.vocab" 
    ]
```

Once you modify the spec, you can update the pipeline by running:

```
$ pachctl update-pipeline -f train.json --reprocess
```

Pachyderm will then automatically kick off a new job to retrain our model with the updated parameters:

```
$ pachctl list-job
ID                                   OUTPUT COMMIT                              STARTED       DURATION   RESTART PROGRESS  DL       UL       STATE
1f62533a-6fe3-4a99-8ec7-e71bebaf0563 model/-                                    2 seconds ago -          0       0 + 0 / 1 0B       0B       running
b5e9e4ba-93ef-4cfa-841b-cff3cbf21529 inference/fe985f6564da4edb9dd04892a911e7af 4 minutes ago 2 seconds  0       2 + 0 / 2 41.85MiB 70B      success
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/7d5eae8ba55c4f04b5a97e117eb93310     9 minutes ago 14 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
```

Not only that, once the model is retrained, Pachyderm see the new model and automatically update our inferences with the latest version of the model.

```
$ pachctl list-job
ID                                   OUTPUT COMMIT                              STARTED        DURATION   RESTART PROGRESS  DL       UL       STATE
5b227d47-c801-4fec-a0c5-8f06732ff4a9 inference/f828b13e7fd54e149101fed150e58d1e 43 seconds ago 1 second   0       2 + 0 / 2 41.85MiB 68B      success
1f62533a-6fe3-4a99-8ec7-e71bebaf0563 model/92c33d731e2b4fc5ab0c6f91e13cce4a     56 seconds ago 3 minutes  0       1 + 0 / 1 977.6KiB 20.93MiB success
b5e9e4ba-93ef-4cfa-841b-cff3cbf21529 inference/fe985f6564da4edb9dd04892a911e7af 5 minutes ago  2 seconds  0       2 + 0 / 2 41.85MiB 70B      success
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/7d5eae8ba55c4f04b5a97e117eb93310     10 minutes ago 14 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
```

### 12. Update the training data set

Let's say that one or more observations in our training data set were corrupt or unwanted.  Thus, we want to update our training data set.  To simulate this, go ahead and open up `labeledTrainData.tsv` (e.g., with `vim`) and remove a couple of the reviews (i.e., the non-header rows).  Then, let's replace our training set:

```
$ vim labeledTrainData.tsv
$ pachctl put-file training master -c -o -f labeledTrainData.tsv
```

Immediately, Pachyderm "knows" that the data has been updated, and it starts a new job to update the model and inferences:

```
$ pachctl list-job
ID                                   OUTPUT COMMIT                              STARTED        DURATION   RESTART PROGRESS  DL       UL       STATE
491fa992-3070-455e-b371-b5ab96065be3 model/-                                    8 seconds ago  -          0       0 + 0 / 1 0B       0B       running
5b227d47-c801-4fec-a0c5-8f06732ff4a9 inference/f828b13e7fd54e149101fed150e58d1e 3 minutes ago  1 second   0       2 + 0 / 2 41.85MiB 68B      success
1f62533a-6fe3-4a99-8ec7-e71bebaf0563 model/92c33d731e2b4fc5ab0c6f91e13cce4a     3 minutes ago  12 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
b5e9e4ba-93ef-4cfa-841b-cff3cbf21529 inference/fe985f6564da4edb9dd04892a911e7af 8 minutes ago  2 seconds  0       2 + 0 / 2 41.85MiB 70B      success
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/7d5eae8ba55c4f04b5a97e117eb93310     12 minutes ago 14 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
```

Not only that, when the new model has been produced, Pachyderm "knows" that there is a new model and updates the previously inferred sentiments:

```
$ pachctl list-job
ID                                   OUTPUT COMMIT                              STARTED            DURATION   RESTART PROGRESS  DL       UL       STATE
8b8fa33d-cbb3-40a6-93b5-6f59407714fb inference/39b2a5d9fe874c0f95b5473af14485a5 About a minute ago 1 second   0       2 + 0 / 2 41.85MiB 70B      success
491fa992-3070-455e-b371-b5ab96065be3 model/960d0e24a836448fae076cf6f2c98e40     About a minute ago 12 seconds 0       1 + 0 / 1 974.3KiB 20.92MiB success
5b227d47-c801-4fec-a0c5-8f06732ff4a9 inference/f828b13e7fd54e149101fed150e58d1e 4 minutes ago      1 second   0       2 + 0 / 2 41.85MiB 68B      success
1f62533a-6fe3-4a99-8ec7-e71bebaf0563 model/92c33d731e2b4fc5ab0c6f91e13cce4a     4 minutes ago      12 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
b5e9e4ba-93ef-4cfa-841b-cff3cbf21529 inference/fe985f6564da4edb9dd04892a911e7af 9 minutes ago      2 seconds  0       2 + 0 / 2 41.85MiB 70B      success
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/7d5eae8ba55c4f04b5a97e117eb93310     13 minutes ago     14 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
```

### 13. Examine pipeline provenance

Let's say that we have updated our model or training set in one of the above scenarios (step 11 or 12).  Now we have multiple inferences that were made with different models and/or training data sets.  How can we know which results came from which specific models and/or training data sets?  This is called "provenance," and Pachyderm gives it to you out of the box.  

Suppose we have run the following jobs:

```
$ pachctl list-job
ID                                   OUTPUT COMMIT                              STARTED            DURATION   RESTART PROGRESS  DL       UL       STATE
8b8fa33d-cbb3-40a6-93b5-6f59407714fb inference/39b2a5d9fe874c0f95b5473af14485a5 About a minute ago 1 second   0       2 + 0 / 2 41.85MiB 70B      success
491fa992-3070-455e-b371-b5ab96065be3 model/960d0e24a836448fae076cf6f2c98e40     About a minute ago 12 seconds 0       1 + 0 / 1 974.3KiB 20.92MiB success
5b227d47-c801-4fec-a0c5-8f06732ff4a9 inference/f828b13e7fd54e149101fed150e58d1e 4 minutes ago      1 second   0       2 + 0 / 2 41.85MiB 68B      success
1f62533a-6fe3-4a99-8ec7-e71bebaf0563 model/92c33d731e2b4fc5ab0c6f91e13cce4a     4 minutes ago      12 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
b5e9e4ba-93ef-4cfa-841b-cff3cbf21529 inference/fe985f6564da4edb9dd04892a911e7af 9 minutes ago      2 seconds  0       2 + 0 / 2 41.85MiB 70B      success
df48056d-bbc2-4fc4-995e-3fae9dc1a659 model/7d5eae8ba55c4f04b5a97e117eb93310     13 minutes ago     14 seconds 0       1 + 0 / 1 977.6KiB 20.93MiB success
```

If we want to know which model and training data set was used for the latest inference, commit id `39b2a5d9fe874c0f95b5473af14485a5`, we just need to inspect the particular commit:

```
$ pachctl inspect-commit inference 39b2a5d9fe874c0f95b5473af14485a5
Commit: inference/39b2a5d9fe874c0f95b5473af14485a5
Parent: f828b13e7fd54e149101fed150e58d1e
Started: 2 minutes ago
Finished: 2 minutes ago
Size: 70B
Provenance:  training/081d10dc1d9346b5872361e0f8d5ecb9  model/960d0e24a836448fae076cf6f2c98e40  reviews/8847b746d8ee441aa69a20fcfae6db2b
```

The `Provenance` tells us exactly which model and training set was used (along with which commit to reviews triggered the sentiment analysis).  For example, if we wanted to see the exact model used, we would just need to reference commit `960d0e24a836448fae076cf6f2c98e40` to the `model` repo:

```
$ pachctl list-file model 960d0e24a836448fae076cf6f2c98e40
NAME                TYPE                SIZE                
imdb.p              file                20.54 MiB           
imdb.vocab          file                392.6 KiB
```

We could get this model to examine it, rerun it, revert to a different model, etc.

## Resources

Pachyderm:

- Join the [Pachyderm Slack team](http://slack.pachyderm.io/) to ask questions, get help, and talk about production deploys.
- Follow [Pachyderm on Twitter](https://twitter.com/pachydermIO), 
- Find [Pachyderm on GitHub](https://github.com/pachyderm/pachyderm), and
- [Spin up Pachyderm](http://docs.pachyderm.io/en/latest/getting_started/getting_started.html) in just a few commands to try this and [other examples](http://docs.pachyderm.io/en/latest/examples/readme.html) locally.
