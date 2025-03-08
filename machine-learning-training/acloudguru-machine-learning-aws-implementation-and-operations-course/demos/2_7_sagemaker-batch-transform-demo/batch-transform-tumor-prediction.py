#!/usr/bin/env python
# coding: utf-8

# # Amazon SageMaker Batch Transform Demo

# _**Use SageMaker's XGBoost to train a binary classification model and for a list of tumors in batch file, predict if each is malignant**_
# 
# Based on AWS sample located at: https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker_batch_transform/batch_transform_associate_predictions_with_input
# ## Setup
# 
# After installing the SageMaker Python SDK
# specify:
# 
# * The SageMaker role arn which has the SageMakerFullAccess policy attached
# * The S3 bucket to use for training and storing model objects.

# In[ ]:


get_ipython().system('pip3 install -U sagemaker')


# In[1]:


import os
import boto3
import sagemaker

role = sagemaker.get_execution_role()
sess = sagemaker.Session()
region = sess.boto_region_name

bucket = sess.default_bucket()
prefix = "DEMO-breast-cancer-prediction-xgboost-highlevel"


# ---
# ## Data sources
# 
# > Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
# 
# ## Data preparation
# 
# Download the data and save it in a local folder with the name data.csv

# In[2]:


import pandas as pd
import numpy as np

s3 = boto3.client("s3")

filename = "wdbc.csv"
s3.download_file(
    f"sagemaker-example-files-prod-{region}", "datasets/tabular/breast_cancer/wdbc.csv", filename
)
data = pd.read_csv(filename, header=None)

# specify columns extracted from wbdc.names
data.columns = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

# save the data
data.to_csv("data.csv", sep=",", index=False)

data.sample(8)


# #### Note:
# * The first field is an 'id' attribute that we'll remove before batch inference since it is not useful for inference
# * The second field, 'diagnosis', uses 'M' for Malignant and 'B'for Benign.
# * There are 30 other numeric features that will be use for training and inferenc.

# Replace the M/B diagnosis with a 1/0 boolean value. 

# In[3]:


data["diagnosis"] = data["diagnosis"].apply(lambda x: ((x == "M")) + 0)
data.sample(8)


# Split the data as follows: 
# 80% for training 
# 10% for validation 
# 10% for batch inference job
# 
# In addition, remove the 'id' field from the training set and validation set as 'id' is not a training feature. 
# Remove the diagnosis attribute for the batch set because this is what we want to predict.

# In[4]:


# data split in three sets, training, validation and batch inference
rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
batch_list = rand_split >= 0.9

data_train = data[train_list].drop(["id"], axis=1)
data_val = data[val_list].drop(["id"], axis=1)
data_batch = data[batch_list].drop(["diagnosis"], axis=1)
data_batch_noID = data_batch.drop(["id"], axis=1)


# Upload the data sets to S3

# In[5]:


train_file = "train_data.csv"
data_train.to_csv(train_file, index=False, header=False)
sess.upload_data(train_file, key_prefix="{}/train".format(prefix))

validation_file = "validation_data.csv"
data_val.to_csv(validation_file, index=False, header=False)
sess.upload_data(validation_file, key_prefix="{}/validation".format(prefix))

batch_file = "batch_data.csv"
data_batch.to_csv(batch_file, index=False, header=False)
sess.upload_data(batch_file, key_prefix="{}/batch".format(prefix))

batch_file_noID = "batch_data_noID.csv"
data_batch_noID.to_csv(batch_file_noID, index=False, header=False)
sess.upload_data(batch_file_noID, key_prefix="{}/batch".format(prefix))


# ---
# 
# ## Training job and model creation

# Start the training job using both training set and validation set. 
# 
# The model will output a probability between 0 and 1 which is predicting the probability of a tumor being malignant.

# In[6]:


get_ipython().run_cell_magic('time', '', 'from time import gmtime, strftime\n\njob_name = "xgb-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\noutput_location = "s3://{}/{}/output/{}".format(bucket, prefix, job_name)\nimage = sagemaker.image_uris.retrieve(\n    framework="xgboost", region=boto3.Session().region_name, version="1.7-1"\n)\n\nsm_estimator = sagemaker.estimator.Estimator(\n    image,\n    role,\n    instance_count=1,\n    instance_type="ml.m5.large",\n    volume_size=50,\n    input_mode="File",\n    output_path=output_location,\n    sagemaker_session=sess,\n)\n\nsm_estimator.set_hyperparameters(\n    objective="binary:logistic",\n    max_depth=5,\n    eta=0.2,\n    gamma=4,\n    min_child_weight=6,\n    subsample=0.8,\n    verbosity=0,\n    num_round=100,\n)\n\ntrain_data = sagemaker.inputs.TrainingInput(\n    "s3://{}/{}/train".format(bucket, prefix),\n    distribution="FullyReplicated",\n    content_type="text/csv",\n    s3_data_type="S3Prefix",\n)\nvalidation_data = sagemaker.inputs.TrainingInput(\n    "s3://{}/{}/validation".format(bucket, prefix),\n    distribution="FullyReplicated",\n    content_type="text/csv",\n    s3_data_type="S3Prefix",\n)\ndata_channels = {"train": train_data, "validation": validation_data}\n\n# Start training by calling the fit method in the estimator\nsm_estimator.fit(inputs=data_channels, logs=True)\n')


# ---
# 
# ## Batch Transform
# Instead of deploying an endpoint and running real-time inference, we'll use SageMaker Batch Transform to run inference on an entire data set in one operation. 
# 

# #### 1. Create a transform job 
# 

# In[8]:


get_ipython().run_cell_magic('time', '', '\nsm_transformer = sm_estimator.transformer(1, "ml.m5.large")\n\n# start a transform job\ninput_location = "s3://{}/{}/batch/{}".format(\n    bucket, prefix, batch_file_noID\n)  # use input data without ID column\nsm_transformer.transform(input_location, content_type="text/csv", split_type="Line")\nsm_transformer.wait()\n')


# Check the output of the Batch Transform job. It should show the list of probabilities of tumors being malignant.

# In[ ]:


import re


def get_csv_output_from_s3(s3uri, batch_file):
    file_name = "{}.out".format(batch_file)
    match = re.match("s3://([^/]+)/(.*)", "{}/{}".format(s3uri, file_name))
    output_bucket, output_prefix = match.group(1), match.group(2)
    s3.download_file(output_bucket, output_prefix, file_name)
    return pd.read_csv(file_name, sep=",", header=None)


# In[ ]:


output_df = get_csv_output_from_s3(sm_transformer.output_path, batch_file_noID)
output_df.head(8)


# #### 2. Join the input and the prediction results 
# 
# We can use batch transform to perform a different transform job to join our original data,
# with our results to get the ID field back. 
# 
# Associate the prediction results with their corresponding input records. We can  use the __input_filter__ to exclude the ID column easily and there's no need to have a separate file in S3.
# 
# * Set __input_filter__ to "$[1:]": indicates that we are excluding column 0 (the 'ID') before processing the inferences and keeping everything from column 1 to the last column (all the features or predictors)  
#   
#   
# * Set __join_source__ to "Input": indicates our desire to join the input data with the inference results  
# 
# * Set __output_filter__ to default "$[1:]", indicating that when presenting the output, we only want to keep column 0 (the 'ID') and the last column (the inference result)

# In[ ]:


# content_type / accept and split_type / assemble_with are required to use IO joining feature
sm_transformer.assemble_with = "Line"
sm_transformer.accept = "text/csv"

# start a transform job
input_location = "s3://{}/{}/batch/{}".format(
    bucket, prefix, batch_file
)  # use input data with ID column cause InputFilter will filter it out
sm_transformer.transform(
    input_location,
    split_type="Line",
    content_type="text/csv",
    input_filter="$[1:]",
    join_source="Input",
    output_filter="$[0,-1]",
)
sm_transformer.wait()


# Check the output of the Batch Transform job in S3. It should show the list of probabilities along with the record ID.

# In[ ]:


output_df = get_csv_output_from_s3(sm_transformer.output_path, batch_file)
output_df.head(8)


# ## Clean up
# In the AWS console, we can see that a model has been created, S3 buckets, and batch transform jobs, however no SageMaker endpoint has been created.
# 
# To avoid unnecessary charges, be sure to delete:
# - S3 buckets
# - Model
# - Jupter notebook
