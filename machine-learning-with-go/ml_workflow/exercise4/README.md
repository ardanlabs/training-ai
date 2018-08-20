# Managing, updating, and scaling the Pachyderm/k8s pipeline

You may not get to all of these bonus exercises during the workshop time, but you can perform these and all of the above steps any time you like with a [simple local Pachyderm install](http://docs.pachyderm.io/en/latest/getting_started/local_installation.html).  You can spin up this local version of Pachyderm is just a few commands and experiment with this, [other Pachyderm examples](http://docs.pachyderm.io/en/latest/examples/readme.html), and/or your own pipelines.

##  1. Parallelize the inference

You may have noticed that our pipeline specs included a `parallelism_spec` field.  This tells Pachyderm how to parallelize a particular pipeline stage. Let's say that in production we start receiving a huge number of patient attributes (or we need to perform periodic batch updating of predictions). In particular, let's say we want to spin up 10 inference workers to perform inference in parallel.

This actually doesn't require any change to our code.  We can simply change our `parallelism_spec` to:

```
  "parallelism_spec": {
    "constant": "10"
  },
```

and update the pipeline:

```
$ vim exercise3/infer.json 
$ pachctl update-pipeline -f exercise3/infer.json 
```

Pachyderm will then spin up 10 inference workers, each running our same `infer` program, to perform inference in parallel.  This can be confirmed by examining the cluster pods:

```
$ kubectl get pods
NAME                               READY     STATUS     RESTARTS   AGE
dash-5d974d8668-wqv29              2/2       Running    0          1h
etcd-66858555cd-cvv5l              1/1       Running    0          1h
pachd-586cd85d4b-kdpkn             1/1       Running    0          1h
pipeline-infer-v2-57jpx            0/2       Pending    0          4s
pipeline-infer-v2-b5khc            0/2       Pending    0          4s
pipeline-infer-v2-bjr2d            0/2       Pending    0          4s
pipeline-infer-v2-gcfh2            0/2       Init:0/1   0          4s
pipeline-infer-v2-jq24k            0/2       Init:0/1   0          4s
pipeline-infer-v2-pjlcr            0/2       Init:0/1   0          4s
pipeline-infer-v2-qntrh            0/2       Init:0/1   0          4s
pipeline-infer-v2-tpk8r            0/2       Init:0/1   0          4s
pipeline-infer-v2-vnsbl            0/2       Pending    0          4s
pipeline-infer-v2-wzm9q            0/2       Pending    0          4s
pipeline-model-v1-2qzjf            2/2       Running    0          24m
pipeline-qc-pre-process-v1-srmnn   2/2       Running    0          16m
pipeline-qcontrol-v1-9j6v9         2/2       Running    0          16m
$ kubectl get pods
NAME                               READY     STATUS    RESTARTS   AGE
dash-5d974d8668-wqv29              2/2       Running   0          1h
etcd-66858555cd-cvv5l              1/1       Running   0          1h
pachd-586cd85d4b-kdpkn             1/1       Running   0          1h
pipeline-infer-v2-57jpx            0/2       Pending   0          1m
pipeline-infer-v2-b5khc            0/2       Pending   0          1m
pipeline-infer-v2-bjr2d            0/2       Pending   0          1m
pipeline-infer-v2-gcfh2            2/2       Running   0          1m
pipeline-infer-v2-jq24k            2/2       Running   0          1m
pipeline-infer-v2-pjlcr            2/2       Running   0          1m
pipeline-infer-v2-qntrh            2/2       Running   0          1m
pipeline-infer-v2-tpk8r            2/2       Running   0          1m
pipeline-infer-v2-vnsbl            0/2       Pending   0          1m
pipeline-infer-v2-wzm9q            0/2       Pending   0          1m
pipeline-model-v1-2qzjf            2/2       Running   0          25m
pipeline-qc-pre-process-v1-srmnn   2/2       Running   0          17m
pipeline-qcontrol-v1-9j6v9         2/2       Running   0          17m 
```

## 2. Update the training data set

Let's say that one or more observations in our training data set were corrupt or unwanted.  Thus, we want to update our training data set.  To simulate this, go ahead and open up `training_prod.csv` (e.g., with `vim`) and remove a couple of the observations (from the non-header rows).  Then, let's replace our training set:

```
$ vim labeledTrainData.tsv
$ pachctl put-file training_data master training_prod.csv -o -f data/training_prod.csv
```

Immediately, Pachyderm "knows" that the data has been updated, and it starts a new job to update the modeling and QC:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                   STARTED            DURATION           RESTART PROGRESS  DL       UL       STATE
e703fbf87fa64c5f85d2aa92bfe21aeb infer/64ad489e235c4b7aa20cb9ee17440582          7 seconds ago      3 seconds          0       2 + 0 / 2 805B     443B     success
95ffa9892cdc467e8c883beef607f50e qcontrol/3562c1d88c1d4447acc78f081937e492       7 seconds ago      1 second           0       1 + 0 / 1 27.99KiB 236B     success
4a118b08e0c54b02a2d171b5e9181955 model/63b05283dfb14ff985fb90b977354f64          7 seconds ago      Less than a second 0       1 + 0 / 1 40.65KiB 207B     success
e50dd8f302b34714b419380031ff76ab infer/2477107b608d452482f2191730e527ba          About a minute ago 7 seconds          0       2 + 0 / 2 805B     443B     success
e5f3074e730548f4ad76503596764c5c qcontrol/e028fdb44454443b8a04abe6ebfb62a5       7 minutes ago      16 seconds         0       1 + 0 / 1 27.99KiB 236B     success
def45a07da9c4157a503ec8c657d99b3 qc_pre_process/17d387a9cfa140b0a3801f99ed5fb63d 7 minutes ago      16 seconds         0       1 + 0 / 1 4.586KiB 27.78KiB success
60e3086e7d9f4f50b50cc8a49394d4a2 model/bb0a351c6975476abe8643dd5448c2e4          7 minutes ago      18 seconds         0       1 + 0 / 1 40.97KiB 207B     success
```

Not only that, when the new model has been produced and QC'ed, Pachyderm "knows" that there is a new model and updates the previously inferred disease predictions.

## 3. Examine pipeline provenance

Let's say that we have updated our model or training set in one of the above scenarios.  Now we have multiple inferences that were made with different models and/or training data sets.  How can we know which results came from which specific models and/or training data sets?  This is called "provenance," and Pachyderm gives it to you out of the box.  

Suppose we have run the following jobs:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                   STARTED            DURATION           RESTART PROGRESS  DL       UL       STATE
e703fbf87fa64c5f85d2aa92bfe21aeb infer/64ad489e235c4b7aa20cb9ee17440582          7 seconds ago      3 seconds          0       2 + 0 / 2 805B     443B     success
95ffa9892cdc467e8c883beef607f50e qcontrol/3562c1d88c1d4447acc78f081937e492       7 seconds ago      1 second           0       1 + 0 / 1 27.99KiB 236B     success
4a118b08e0c54b02a2d171b5e9181955 model/63b05283dfb14ff985fb90b977354f64          7 seconds ago      Less than a second 0       1 + 0 / 1 40.65KiB 207B     success
e50dd8f302b34714b419380031ff76ab infer/2477107b608d452482f2191730e527ba          About a minute ago 7 seconds          0       2 + 0 / 2 805B     443B     success
e5f3074e730548f4ad76503596764c5c qcontrol/e028fdb44454443b8a04abe6ebfb62a5       7 minutes ago      16 seconds         0       1 + 0 / 1 27.99KiB 236B     success
def45a07da9c4157a503ec8c657d99b3 qc_pre_process/17d387a9cfa140b0a3801f99ed5fb63d 7 minutes ago      16 seconds         0       1 + 0 / 1 4.586KiB 27.78KiB success
60e3086e7d9f4f50b50cc8a49394d4a2 model/bb0a351c6975476abe8643dd5448c2e4          7 minutes ago      18 seconds         0       1 + 0 / 1 40.97KiB 207B     success
```

If we want to know which model and training data set was used for the latest inference, commit id `64ad489e235c4b7aa20cb9ee17440582`, we just need to inspect the particular commit:

```
$ pachctl inspect-commit infer 64ad489e235c4b7aa20cb9ee17440582
Commit: infer/64ad489e235c4b7aa20cb9ee17440582
Parent: 2477107b608d452482f2191730e527ba
Started: 2 minutes ago
Finished: About a minute ago
Size: 443B
Provenance:  qc_data/ee45762175b94b9782ad5ad3a5eabc32  __spec__/bd864170f9c44797a9d5022f207d657d  __spec__/4924bf0c685d4d77a3cc60dda008cb02  __spec__/d88e4080ab1644a29de15277516725e3  attributes/cf65b8fe85184d80bff6976a6426f5b2  model/63b05283dfb14ff985fb90b977354f64  __spec__/b022c5ba0ada422ea97f7db8f960680d  qc_pre_process/17d387a9cfa140b0a3801f99ed5fb63d  qcontrol/3562c1d88c1d4447acc78f081937e492  training_data/e452c4c44c914d5187de47edade656b1
```

The `Provenance` tells us exactly which model and training set was used (along with which commit to `attributes` triggered the inference).  For example, if we wanted to see the exact model used, we would just need to reference commit `63b05283dfb14ff985fb90b977354f64` to the `model` repo:

```
$ pachctl get-file model 63b05283dfb14ff985fb90b977354f64 model.json
{
  "intercept": 152.17757154861354,
  "coefficients": [
    {
      "name": "bmi",
      "coefficient": 676.7442087119301
    },
    {
      "name": "ltg",
      "coefficient": 594.2267311960092
    }
  ]
}
```

We could get this model to examine it, rerun it, revert to a different model, etc.
