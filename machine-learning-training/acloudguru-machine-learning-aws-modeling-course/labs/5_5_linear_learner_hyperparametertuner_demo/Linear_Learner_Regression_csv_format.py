#!/usr/bin/env python
# coding: utf-8

# ## Regression with Amazon SageMaker Linear Learner algorithm
# 

# ---
# 
# This notebook's CI test result for us-west-2 is as follows. CI test results in other regions can be found at the end of the notebook. 
# 
# ![This us-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-2/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ---

# _**Single machine training for regression with Amazon SageMaker Linear Learner algorithm**_
# 
# ---
# 
# ---
# ## Contents
# 1. [Introduction](#Introduction)
# 2. [Setup](#Setup)
#    1. [Exploring the dataset](#Exploring-the-dataset)
# 3. [Training the Linear Learner Model](#Training-the-linear-learner-model)
#    1. [Training with SageMaker Training](#Training-with-sagemaker-training)
#    2. [Training with Automatic Model Tuning (HPO)](#Training-with-automatic-model-tuning-HPO)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
# 5. [Inference](#Inference)
# 6. [Delete the Endpoint](#Delete-the-Endpoint)
# 7. [Appendix](#Appendix)
#   1. [Downloading the dataset](#Downloading-the-dataset)
#   2. [libsvm to csv convertion](#libsvm-to-csv-convertion)
#   3. [Dividing the data](#Dividing-the-data)
#   4. [Data Ingestion](#Data-ingestion)
# ---
# ## Introduction
# 
# This notebook demonstrates the use of Amazon SageMaker’s implementation of the Linear Learner algorithm to train and host a regression model. We use the [Abalone data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html) originally from the [UCI data repository](https://archive.ics.uci.edu/ml/datasets/abalone). 
# 
# The dataset contains 9 fields, starting with the Rings number which is a number indicating the age of the abalone (as age equals to number of rings plus 1.5). Usually the number of rings are counted through microscopes to estimate the abalone's age. So we will use our algorithm to predict the abalone age based on the other features which are mentioned respectively as below within the dataset. 
# 
# 'Rings','sex','Length','Diameter','Height','Whole Weight','Shucked Weight','Viscera Weight' and 'Shell Weight'
# 
# The above features starting from sex to Shell.weight are physical measurements that can be measured using the correct tools, so we improve the complixety of having to examine the abalone under microscopes to understand it's age.
# 

# ---
# ## Setup
# 
# 
# This notebook was tested in Amazon SageMaker Studio on a ml.t3.medium instance with Python 3 (Data Science) kernel.
# 
# Let's start by specifying:
# 1. The S3 buckets and prefixes that you want to use for training data and model data. This should be within the same region as the Notebook Instance, training, and hosting.
# 1. The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these. Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).

# In[ ]:


get_ipython().system(' pip install --upgrade sagemaker')


# In[ ]:


import os
import boto3
import re
import sagemaker


role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# S3 bucket for training data.
# Feel free to specify a different bucket and prefix.
data_bucket = f"sagemaker-example-files-prod-{region}"
data_prefix = "datasets/tabular/uci_abalone"


# S3 bucket for saving code and model artifacts.
# Feel free to specify a different bucket and prefix
output_bucket = sagemaker.Session().default_bucket()
output_prefix = "sagemaker/DEMO-linear-learner-abalone-regression"


# ## Exploring the dataset
# 
# We pre-processed the Abalone dataset [1] and stored in a S3 bucket. It was downloaded from the [National Taiwan University's CS department's tools for regression on the abalone dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone). Scripts used in downloading and pre-processing can be found in the [Appendix](#Appendix). These include downloading data, converting data from libsvm format to csv format, dividing it into train, validation and test and uploading it to S3 bucket. 
# 
# The dataset contains a total of 9 fields. Throughout this notebook, they will be named as follows 'age','sex','Length','Diameter','Height','Whole.weight','Shucked.weight','Viscera.weight' and 'Shell.weight' respectively.
# 
# The below data frame representation explain the value of each field.
# Note: the age field is in integer representation and the rest of the features are in the format of "feature_number":"feature_value"
# 
# ```
# **'data.frame'**:
# age              int  15 7 9 10 7 8 20 16 9 19 ...
# Sex               <feature_number>: Factor w/ 3 levels "F","I","M" values of 1,2,3
# Length            <feature_number>: float  0.455 0.35 0.53 0.44 0.33 0.425 ...
# Diameter          <feature_number>: float  0.365 0.265 0.42 0.365 0.255 0.3 ...
# Height            <feature_number>: float  0.095 0.09 0.135 0.125 0.08 0.095 ...
# Whole.weight      <feature_number>: float  0.514 0.226 0.677 0.516 0.205 ...
# Shucked.weight    <feature_number>: float  0.2245 0.0995 0.2565 0.2155 0.0895 ...
# Viscera.weight    <feature_number>: float  0.101 0.0485 0.1415 0.114 0.0395 ...
# Shell.weight      <feature_number>: float  0.15 0.07 0.21 0.155 0.055 0.12 ...
# ```
# >[1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

# In[ ]:


import boto3

s3 = boto3.client("s3")

FILE_TRAIN = "abalone_dataset1_train.csv"
FILE_TEST = "abalone_dataset1_test.csv"
FILE_VALIDATION = "abalone_dataset1_validation.csv"

# downloading the train, test, and validation files from data_bucket
s3.download_file(data_bucket, f"{data_prefix}/train_csv/{FILE_TRAIN}", FILE_TRAIN)
s3.download_file(data_bucket, f"{data_prefix}/test_csv/{FILE_TEST}", FILE_TEST)
s3.download_file(data_bucket, f"{data_prefix}/validation_csv/{FILE_VALIDATION}", FILE_VALIDATION)
s3.upload_file(FILE_TRAIN, output_bucket, f"{output_prefix}/train/{FILE_TRAIN}")
s3.upload_file(FILE_TEST, output_bucket, f"{output_prefix}/test/{FILE_TEST}")
s3.upload_file(FILE_VALIDATION, output_bucket, f"{output_prefix}/validation/{FILE_VALIDATION}")


# In[ ]:


import pandas as pd  # Read in csv and store in a pandas dataframe

df = pd.read_csv(
    FILE_TRAIN,
    sep=",",
    encoding="latin1",
    names=[
        "age",
        "sex",
        "Length",
        "Diameter",
        "Height",
        "Whole.weight",
        "Shucked.weight",
        "Viscera.weight",
        "Shell.weight",
    ],
)
print(df.head(1))


# 
# ---
# Let us prepare the handshake between our data channels and the algorithm. To do this, we need to create the `sagemaker.session.s3_input` objects from our [data channels](https://sagemaker.readthedocs.io/en/v1.2.4/session.html#). These objects are then put in a simple dictionary, which the algorithm consumes. Notice that here we use a `content_type` as `text/csv` for the pre-processed file in the data_bucket. We use two channels here one for training and the second one for validation. The testing samples from above will be used on the prediction step.

# In[ ]:


# creating the inputs for the fit() function with the training and validation location
s3_train_data = f"s3://{output_bucket}/{output_prefix}/train"
print(f"training files will be taken from: {s3_train_data}")
s3_validation_data = f"s3://{output_bucket}/{output_prefix}/validation"
print(f"validation files will be taken from: {s3_validation_data}")
output_location = f"s3://{output_bucket}/{output_prefix}/output"
print(f"training artifacts output location: {output_location}")

# generating the session.s3_input() format for fit() accepted by the sdk
train_data = sagemaker.inputs.TrainingInput(
    s3_train_data,
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None,
)
validation_data = sagemaker.inputs.TrainingInput(
    s3_validation_data,
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None,
)


# ## Training the Linear Learner Model
# 
# Training can be done by either calling SageMaker Training with a set of hyperparameters values to train with, or by leveraging SageMaker Automatic Model Tuning ([AMT](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)). AMT, also known as hyperparameter tuning (HPO), finds the best version of a model by running many training jobs on your dataset using the algorithm and ranges of hyperparameters that you specify. It then chooses the hyperparameter values that result in a model that performs the best, as measured by a metric that you choose.
# 
# In this notebook, both methods are used for demonstration purposes, but the model that the HPO job creates is the one that is eventually hosted. You can instead choose to deploy the model created by the standalone training job by changing the below variable `deploy_amt_model` to False.
# 
# ### Training with SageMaker Training
# 
# First, we retrieve the image for the Linear Learner Algorithm according to the region.

# In[ ]:


# getting the linear learner image according to the region
from sagemaker.image_uris import retrieve

container = retrieve("linear-learner", boto3.Session().region_name, version="1")
print(container)
deploy_amt_model = True


# Then we create an [estimator from the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) using the Linear Learner container image and we setup the training parameters and hyperparameters configuration.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import boto3\nimport sagemaker\nfrom time import gmtime, strftime\n\nsess = sagemaker.Session()\n\njob_name = "DEMO-linear-learner-abalone-regression-" + strftime("%Y%m%d-%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\nlinear = sagemaker.estimator.Estimator(\n    container,\n    role,\n    input_mode="File",\n    instance_count=1,\n    instance_type="ml.m4.xlarge",\n    output_path=output_location,\n    sagemaker_session=sess,\n)\n\nlinear.set_hyperparameters(\n    feature_dim=8,\n    epochs=16,\n    wd=0.01,\n    loss="absolute_loss",\n    predictor_type="regressor",\n    normalize_data=True,\n    optimizer="adam",\n    mini_batch_size=100,\n    lr_scheduler_step=100,\n    lr_scheduler_factor=0.99,\n    lr_scheduler_minimum_lr=0.0001,\n    learning_rate=0.1,\n)\n')


# ---
# After configuring the Estimator object and setting the hyperparameters for this object. The only remaining thing to do is to train the algorithm. The following cell will train the algorithm. Training the algorithm involves a few steps. Firstly, the instances that we requested while creating the Estimator classes are provisioned and are setup with the appropriate libraries. Then, the data from our channels are downloaded into the instance. Once this is done, the training job begins. The provisioning and data downloading will take time, depending on the size of the data. Therefore it might be a few minutes before we start getting data logs for our training jobs. The data logs will also print out Mean Average Precision (mAP) on the validation data, among other losses, for every run of the dataset once or one epoch. This metric is a proxy for the quality of the algorithm.
# 
# Once the job has finished a "Job complete" message will be printed. The trained model can be found in the S3 bucket that was setup as output_path in the estimator. For this example,the training time takes between 4 and 6 minutes.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'linear.fit(inputs={"train": train_data, "validation": validation_data}, job_name=job_name)\n')


# ### Training with Automatic Model Tuning ([HPO](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)) <a id='AMT'></a>
# ***
# As mentioned above, instead of manually configuring our hyper parameter values and training with SageMaker Training, we'll use Amazon SageMaker Automatic Model Tuning. 
#         
# The code sample below shows you how to use the HyperParameterTuner. For recommended default hyparameter ranges, check the [Amazon SageMaker Linear Learner HPs documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html). 
# 
# The tuning job will take 8 to 10 minutes to complete.
# ***

# In[ ]:


import time
from sagemaker.tuner import IntegerParameter, ContinuousParameter
from sagemaker.tuner import HyperparameterTuner

job_name = "DEMO-ll-aba-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print("Tuning job name: ", job_name)

# Linear Learner tunable hyper parameters can be found here https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner-tuning.html
hyperparameter_ranges = {
    "wd": ContinuousParameter(1e-7, 1, scaling_type="Auto"),
    "learning_rate": ContinuousParameter(1e-5, 1, scaling_type="Auto"),
    "mini_batch_size": IntegerParameter(100, 2000, scaling_type="Auto"),
}

# Increase the total number of training jobs run by AMT, for increased accuracy (and training time).
max_jobs = 6
# Change parallel training jobs run by AMT to reduce total training time, constrained by your account limits.
# if max_jobs=max_parallel_jobs then Bayesian search turns to Random.
max_parallel_jobs = 2


hp_tuner = HyperparameterTuner(
    linear,
    "validation:mse",
    hyperparameter_ranges,
    max_jobs=max_jobs,
    max_parallel_jobs=max_parallel_jobs,
    objective_type="Minimize",
)


# Launch a SageMaker Tuning job to search for the best hyperparameters
hp_tuner.fit(inputs={"train": train_data, "validation": validation_data}, job_name=job_name)


# ## Set up hosting for the model
# 
# Once the training is done, we can deploy the trained model as an Amazon SageMaker real-time hosted endpoint. This will allow us to make predictions (or inference) from the model. Note that we don't have to host on the same instance (or type of instance) that we used to train. Training is a prolonged and compute heavy job that require a different of compute and memory requirements that hosting typically do not. We can choose any type of instance we want to host the model. In our case we chose the ml.m4.xlarge instance to train, but we choose to host the model on the less expensive cpu instance, ml.c4.xlarge. The endpoint deployment can be accomplished as follows:
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '# creating the endpoint out of the trained model\n\nif deploy_amt_model:\n    linear_predictor = hp_tuner.deploy(initial_instance_count=1, instance_type="ml.c4.xlarge")\nelse:\n    linear_predictor = linear.deploy(initial_instance_count=1, instance_type="ml.c4.xlarge")\nprint(f"\\ncreated endpoint: {linear_predictor.endpoint_name}")\n')


# ## Inference
# 
# Now that the trained model is deployed at an endpoint that is up-and-running, we can use this endpoint for inference. To do this, we are going to configure the [predictor object](https://sagemaker.readthedocs.io/en/v1.2.4/predictors.html) to parse contents of type text/csv and deserialize the reply received from the endpoint to json format.
# 

# In[ ]:


# configure the predictor to accept to serialize csv input and parse the reposne as json
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

linear_predictor.serializer = CSVSerializer()
linear_predictor.deserializer = JSONDeserializer()


# ---
# We then use the test file containing the records of the data that we kept to test the model prediction. By running below cell multiple times we are selecting random sample from the testing samples to perform inference with.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import json\nfrom itertools import islice\nimport math\nimport struct\nimport boto3\nimport random\n\n# getting testing sample from our test file\ntest_data = [l for l in open(FILE_TEST, "r")]\nsample = random.choice(test_data).split(",")\nactual_age = sample[0]\npayload = sample[1:]  # removing actual age from the sample\npayload = ",".join(map(str, payload))\n\n# Invoke the predicor and analyise the result\nresult = linear_predictor.predict(payload)\n\n# extracting the prediction value\nresult = round(float(result["predictions"][0]["score"]), 2)\n\n\naccuracy = str(round(100 - ((abs(float(result) - float(actual_age)) / float(actual_age)) * 100), 2))\nprint(f"Actual age: {actual_age}\\nPrediction: {result}\\nAccuracy: {accuracy}")\n')


# ## Delete the Endpoint
# Having an endpoint running will incur some costs. Therefore as a clean-up job, we should delete the endpoint.

# In[ ]:


sagemaker.Session().delete_endpoint(linear_predictor.endpoint_name)
print(f"deleted {linear_predictor.endpoint_name} successfully!")


# # Appendix

# ## Downloading the dataset
# 
# We are downloading the dataset in the original libsvm format, more info about this format can be found [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html).

# In[ ]:


get_ipython().run_cell_magic('time', '', 's3 = boto3.client("s3")\n\n# Load the dataset\nSOURCE_DATA = "abalone"\ns3.download_file(data_bucket, f"{data_prefix}/abalone.libsvm", SOURCE_DATA)\n')


# ## libsvm to csv convertion
# Then we convert this dataset into csv format which is one of the accepted formats by the Linear Learner Algorithm, more information [here](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html#ll-input_output).
# 
# The value of the age field is parsed as integer and the value for the features is extracted from the format of "feature_number":"feature_value" to return only the value of the corresponding feature then the final frame is written to the output file.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# import numpy and pandas libraries for working with data\nimport numpy as np\nimport pandas as pd  # Read in csv and store in a pandas dataframe\n\ndf = pd.read_csv(\n    SOURCE_DATA,\n    sep=" ",\n    encoding="latin1",\n    names=[\n        "age",\n        "sex",\n        "Length",\n        "Diameter",\n        "Height",\n        "Whole.weight",\n        "Shucked.weight",\n        "Viscera.weight",\n        "Shell.weight",\n    ],\n)\n\n# converting the age to int value\ndf["age"] = df["age"].astype(int)\n\n# drop any null values\ndf.dropna(inplace=True)\n\n# Extracting the features values from  the libsvm format\nfeatures = [\n    "sex",\n    "Length",\n    "Diameter",\n    "Height",\n    "Whole.weight",\n    "Shucked.weight",\n    "Viscera.weight",\n    "Shell.weight",\n]\nfor feature in features:\n    if feature == "sex":\n        df[feature] = (df[feature].str.split(":", n=1, expand=True)[1]).astype(int)\n    else:\n        df[feature] = (df[feature].str.split(":", n=1, expand=True)[1]).astype(float)\n\n\n# #writing the final data in the correct format\ndf.to_csv("new_data_set_float32.csv", sep=",", index=False, header=None)\n\nprint(df.head(1))\n')


# ## Dividing the data
# 
# Following methods split the data into train/test/validation datasets and upload files to S3.
# 

# In[ ]:


import io
import boto3
import random


def data_split(
    FILE_DATA,
    FILE_TRAIN,
    FILE_VALIDATION,
    FILE_TEST,
    PERCENT_TRAIN,
    PERCENT_VALIDATION,
    PERCENT_TEST,
):
    data = [l for l in open(FILE_DATA, "r")]
    train_file = open(FILE_TRAIN, "w")
    valid_file = open(FILE_VALIDATION, "w")
    tests_file = open(FILE_TEST, "w")

    num_of_data = len(data)
    num_train = int((PERCENT_TRAIN / 100.0) * num_of_data)
    num_valid = int((PERCENT_VALIDATION / 100.0) * num_of_data)
    num_tests = int((PERCENT_TEST / 100.0) * num_of_data)

    data_fractions = [num_train, num_valid, num_tests]
    split_data = [[], [], []]

    rand_data_ind = 0

    for split_ind, fraction in enumerate(data_fractions):
        for i in range(fraction):
            rand_data_ind = random.randint(0, len(data) - 1)
            split_data[split_ind].append(data[rand_data_ind])
            data.pop(rand_data_ind)

    for l in split_data[0]:
        train_file.write(l)

    for l in split_data[1]:
        valid_file.write(l)

    for l in split_data[2]:
        tests_file.write(l)

    train_file.close()
    valid_file.close()
    tests_file.close()


def write_to_s3(fobj, bucket, key):
    return (
        boto3.Session(region_name=region)
        .resource("s3")
        .Bucket(bucket)
        .Object(key)
        .upload_fileobj(fobj)
    )


def upload_to_s3(bucket, prefix, channel, filename):
    fobj = open(filename, "rb")
    key = f"{prefix}/{channel}/{filename}"
    url = f"s3://{bucket}/{key}"
    print(f"Writing to {url}")
    write_to_s3(fobj, bucket, key)


# ### Data ingestion
# 
# Next, we read the dataset from the existing repository into memory, for preprocessing prior to training. This processing could be done *in situ* by Glue, Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Load the dataset\nFILE_DATA = "new_data_set_float32.csv"\n\n# split the downloaded data into train/test/validation files\nFILE_TRAIN = "abalone_dataset1_train.csv"\nFILE_VALIDATION = "abalone_dataset1_validation.csv"\nFILE_TEST = "abalone_dataset1_test.csv"\nPERCENT_TRAIN = 70\nPERCENT_VALIDATION = 15\nPERCENT_TEST = 15\ndata_split(\n    FILE_DATA,\n    FILE_TRAIN,\n    FILE_VALIDATION,\n    FILE_TEST,\n    PERCENT_TRAIN,\n    PERCENT_VALIDATION,\n    PERCENT_TEST,\n)\n\n# S3 bucket to store training data.\n# Feel free to specify a different bucket and prefix.\nbucket = sagemaker.Session().default_bucket()\nprefix = "sagemaker/DEMO-linear-learner-abalone-regression"\n\n# upload the files to the S3 bucket\nupload_to_s3(bucket, prefix, "train", FILE_TRAIN)\nupload_to_s3(bucket, prefix, "validation", FILE_VALIDATION)\nupload_to_s3(bucket, prefix, "test", FILE_TEST)\n')


# ## Notebook CI Test Results
# 
# This notebook was tested in multiple regions. The test results are as follows, except for us-west-2 which is shown at the top of the notebook.
# 
# ![This us-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This us-east-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-2/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This us-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This ca-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ca-central-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This sa-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/sa-east-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This eu-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This eu-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-2/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This eu-west-3 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-3/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This eu-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-central-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This eu-north-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-north-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This ap-southeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This ap-southeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-2/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This ap-northeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This ap-northeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-2/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
# ![This ap-south-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-south-1/introduction_to_amazon_algorithms|linear_learner_abalone|Linear_Learner_Regression_csv_format.ipynb)
# 
