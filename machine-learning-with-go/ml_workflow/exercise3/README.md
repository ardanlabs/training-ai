# Deploying the Pachyderm Pipeline on k8s

This tutorial will walk you through the deployment of our training, quality control, and inference programs as Pachyderm pipeline stages on top of Kubernetes. Once the pipeline is deployed, it will be able to automatically make updates based on new input data, and it will allow us to track and monitor all changes to the model, data, predictions, etc.

This will allow us to:

- Keep a rigorous historical record of exactly what models were used on what data to produce which results.
- Automatically update online ML models when training data or parameterization changes.
- Easily revert to other versions of an ML model when a new model is not performing or when “bad data” is introduced into a training data set.

As a reminder, the general structure of our pipeline will look like this:

![Alt text](https://docs.google.com/drawings/d/e/2PACX-1vSEuTeHVqRTzAmuQBILI_KExrF0oZsWJ_FaclWONTM60E_e5KfNUHAqK8S5L92R4AKjcmMWipkouUYd/pub?w=1537&h=612)

The cylinders represent data “repositories” in which Pachyderm will version training, model, etc. data (think “git for data”).  These data repositories are then input/output of the linked data processing stages (represented by the boxes in the figure).  

## 1. Connect to Pachyderm/k8s

Your workshop instance should already be connected to a running kubernetes cluster with Pachyderm. You can verify this by getting the Pachyderm version with Pachyderm’s `pachctl` CLI tool:

```
$ pachctl version
COMPONENT           VERSION
pachctl             1.7.4
pachd               1.7.4
```

You should also be able to see Pachyderm running in Kubernetes via the `kubectl` CLI tool:

```
$ kubectl get pods
NAME                     READY     STATUS    RESTARTS   AGE
dash-5d974d8668-wqv29    2/2       Running   0          59m
etcd-66858555cd-cvv5l    1/1       Running   0          59m
pachd-586cd85d4b-kdpkn   1/1       Running   0          59m
```

## 2. Create the input data repositories

We will need to create the three input data repositories (for our training data, quality control (i.e., holdout) data, and the input iris attributes).  To do this run:

```
$ pachctl create-repo training_data
$ pachctl create-repo qc_data
$ pachctl create-repo attributes
```

As a sanity check, we can list out the current repos, and you should see the two repos you just created:

```
$ pachctl list-repo
NAME          CREATED        SIZE
attributes    3 seconds ago  0B
qc_data       7 seconds ago  0B
training_data 13 seconds ago 0B
```

## 3. Commit the training and QC data sets into Pachyderm

We have our training data repository, but we haven't put our training data set into this repository yet.  The training data set, `training_prod.csv`,  and the QC data set, `holdout.csv`, were created with in Jupyter and should be saved here in the [data](../data) directory after you run that notebook.

To get this data into Pachyderm:

```
$ cd /home/pachrat/training-ai/machine-learning-with-go/ml_workflow
$ pachctl put-file training_data master training_prod.csv -f data/training_prod.csv
$ pachctl put-file qc_data master holdout.csv -f data/holdout.csv
```

Then, you should be able to see the following:

```
$ pachctl list-repo
NAME          CREATED       SIZE
attributes    3 minutes ago 0B
qc_data       3 minutes ago 4.586KiB
training_data 3 minutes ago 41.28KiB
$ pachctl list-file training_data master
NAME              TYPE SIZE
training_prod.csv file 41.28KiB
$ pachctl list-file qc_data master
NAME        TYPE SIZE
holdout.csv file 4.586KiB
```

## 4. Create the training pipeline

Next, we can create the `model` pipeline stage to process the data in the `training_data` repository. To do this, we just need to provide Pachyderm with a JSON pipeline specification that tells Pachyderm how to process the data. You can utilize the training image that was created in the last lab exercise by using the [model.json](model.json) specification. 

Create the training pipeline as follows:

```
$ pachctl create-pipeline -f exercise3/model.json
```

Immediately you will notice that Pachyderm kicked off a job to perform the model training:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                          STARTED        DURATION  RESTART PROGRESS  DL       UL   STATE
531312dd718546f8a387d5ad7e191bef model/8c0498fd88aa4b4b98ddbd996df9ef87 25 seconds ago 3 seconds 0       1 + 0 / 1 41.28KiB 207B success
```

After the job finishes, your model has successfully been trained on the cluster! You should see:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                          STARTED       DURATION       RESTART PROGRESS  DL       UL       STATE
1a8225537992422f87c8468a16d0718b model/6e7cf823910b4ae68c8d337614654564 2 minutes ago About a minute 0       1 + 0 / 1 4.444KiB 49.86KiB success
$ pachctl list-repo
NAME                CREATED             SIZE
model               2 minutes ago       43.67 KiB
training            8 minutes ago       4.444 KiB
attributes          7 minutes ago       0 B
$ pachctl list-file model master
NAME                TYPE                SIZE
model.jld           file                43.67 KiB
```

## 5. QC our model

Great! We now have a trained model that will infer the progression of the disease diabetes based on input attributes (`bmi` and `ltg`). However, before utilizing this model for production inferences, we should quality control it. We can do that with the data set that we held out from our training/test data, `holdout.csv`. 

First we need to pre-process the CSV file into JSON attribute files like our inference Go program will process.  We will do that with a `qc_pre_process` pipeline. Then, we will calculate an evaluation metric (RMSE) on that pre-processed data. If the evaluation metric is good enough we will pass the trained model through (to the next stage of the pipeline), such that it can be used for predictions. 

Create the QC pre-processing and QC pipelines using [qc_pre_process.json](qc_pre_process.json) and [qcontrol.json](qcontrol.json):

```
$ pachctl create-pipeline -f exercise3/qc_pre_process.json
$ pachctl create-pipeline -f exercise3/qcontrol.json
```

This will automatically run the QC process on the cluster:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                   STARTED        DURATION  RESTART PROGRESS  DL       UL       STATE
83c7e395ea6c4cdc9ba7ce9a77551bad qcontrol/52a471b60ade43cd9b177005349b8d47       6 seconds ago  4 seconds 0       1 + 0 / 1 27.99KiB 236B     success
851d03408c8743b08aa60b381d987e4f qc_pre_process/e327095639f84d4796230e70c18724f0 11 seconds ago 3 seconds 0       1 + 0 / 1 4.586KiB 27.78KiB success
531312dd718546f8a387d5ad7e191bef model/8c0498fd88aa4b4b98ddbd996df9ef87          7 minutes ago  3 seconds 0       1 + 0 / 1 41.28KiB 207B     success
```

To check that our model passed QC (and to see the corresponding RMSE):

```
$ pachctl list-file qcontrol master
NAME       TYPE SIZE
model.json file 236B
$ pachctl get-file qcontrol master model.json
{
  "rmse": 46.44507083538092,
  "intercept": 152.38005026421627,
  "coefficients": [
    {
      "name": "bmi",
      "coefficient": 670.4022409970744
    },
    {
      "name": "ltg",
      "coefficient": 593.7168799545466
    }
  ]
}
```

## 6. Commit input attributes

Great! We now have a trained and QC'ed model.  Let's commit some example patient attributes into Pachyderm that we would like to run through the inference.  We have a couple examples under [test](../data/test).  Feel free to use these, or even create your own.  To commit a couple of these samples:

```
$ pachctl put-file attributes master 1.json -f data/test/1.json
$ pachctl put-file attributes master 2.json -f data/test/2.json
```

You should then see:

```
$ pachctl list-file attributes master
NAME   TYPE SIZE
1.json file 166B
2.json file 167B
```

## 7. Create the inference pipeline

We have another JSON specification, [infer.json](infer.json), that will tell Pachyderm how to perform the processing for the inference stage.  To create the inference stage, we simply run:

```
$ pachctl create-pipeline -f exercise3/infer.json
```

This will immediately kick off an inference job, because we have committed unprocessed attributes into the `attributes` repo.  The results will then be versioned in a corresponding `infer` data repository:

```
$ pachctl list-job
ID                               OUTPUT COMMIT                                   STARTED        DURATION   RESTART PROGRESS  DL       UL       STATE
e50dd8f302b34714b419380031ff76ab infer/2477107b608d452482f2191730e527ba          13 seconds ago 7 seconds  0       2 + 0 / 2 805B     443B     success
e5f3074e730548f4ad76503596764c5c qcontrol/e028fdb44454443b8a04abe6ebfb62a5       5 minutes ago  16 seconds 0       1 + 0 / 1 27.99KiB 236B     success
def45a07da9c4157a503ec8c657d99b3 qc_pre_process/17d387a9cfa140b0a3801f99ed5fb63d 5 minutes ago  16 seconds 0       1 + 0 / 1 4.586KiB 27.78KiB success
60e3086e7d9f4f50b50cc8a49394d4a2 model/bb0a351c6975476abe8643dd5448c2e4          5 minutes ago  18 seconds 0       1 + 0 / 1 40.97KiB 207B     success
$ pachctl list-file infer master
NAME   TYPE SIZE
1.json file 221B
2.json file 222B
```

## 8. Examine the results

We have created results from the inference, but how do we examine those results?  There are multiple ways, but an easy way is to just "get" the specific files out of Pachyderm's data versioning:

```
$ pachctl get-file infer master 1.json
{
  "predicted_diabetes_progression": 205.56129090286984,
  "independent_variables": [
    {
      "name": "bmi",
      "value": 0.0616962065187
    },
    {
      "name": "ltg",
      "value": 0.0199084208763
    }
  ]
}
```