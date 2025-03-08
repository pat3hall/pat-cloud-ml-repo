#!/usr/bin/env python
# coding: utf-8

# # Train a SKLearn Model using Script Mode
# 

# ---
# 
# This notebook's CI test result for us-west-2 is as follows. CI test results in other regions can be found at the end of the notebook. 
# 
# ![This us-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-2/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ---

# 
# The aim of this notebook is to demonstrate how to train and deploy a scikit-learn model in Amazon SageMaker. The method used is called Script Mode, in which we write a script to train our model and submit it to the SageMaker Python SDK. For more information, feel free to read [Using Scikit-learn with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html).
# 
# ## Runtime
# This notebook takes approximately 15 minutes to run.
# 
# ## Contents
# 1. [Download data](#Download-data)
# 1. [Prepare data](#Prepare-data)
# 1. [Train model](#Train-model)
# 1. [Deploy and test endpoint](#Deploy-and-test-endpoint)
# 1. [Cleanup](#Cleanup)

# ## Download data 
# Download the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris), which is the data used to trained the model in this demo.

# In[1]:


get_ipython().system('pip install -U sagemaker')


# In[2]:


import boto3
import pandas as pd
import numpy as np

s3 = boto3.client("s3")
s3.download_file(
    f"sagemaker-example-files-prod-{boto3.session.Session().region_name}",
    "datasets/tabular/iris/iris.data",
    "iris.data",
)

df = pd.read_csv(
    "iris.data", header=None, names=["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
)
df.head()


# ## Prepare data
# Next, we prepare the data for training by first converting the labels from string to integers. Then we split the data into a train dataset (80% of the data) and test dataset (the remaining 20% of the data) before saving them into CSV files. Then, these files are uploaded to S3 where the SageMaker SDK can access and use them to train the model.

# In[3]:


# Convert the three classes from strings to integers in {0,1,2}
df["class_cat"] = df["class"].astype("category").cat.codes
categories_map = dict(enumerate(df["class"].astype("category").cat.categories))
print(categories_map)
df.head()


# In[15]:


df_rand= df.sample(frac=1)
df_rand.head()


# In[16]:


# Split the data into 80-20 train-test split
num_samples = df_rand.shape[0]
split = round(num_samples * 0.8)
train = df_rand.iloc[:split, :]
test = df_rand.iloc[split:, :]
print("{} train, {} test".format(split, num_samples - split))


# In[17]:


# Write train and test CSV files
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)


# In[24]:


# Create a sagemaker session to upload data to S3
import sagemaker

sagemaker_session = sagemaker.Session()

# Upload data to default S3 bucket
prefix = "DEMO-sklearn-iris"
training_input_path = sagemaker_session.upload_data("train.csv", key_prefix=prefix + "/training")


# In[ ]:





# ## Train model
# The model is trained using the SageMaker SDK's Estimator class. Firstly, get the execution role for training. This role allows us to access the S3 bucket in the last step, where the train and test data set is located.

# In[25]:


# Use the current execution role for training. It needs access to S3
role = sagemaker.get_execution_role()
print(role)


# Then, it is time to define the SageMaker SDK Estimator class. We use an Estimator class specifically desgined to train scikit-learn models called `SKLearn`. In this estimator, we define the following parameters:
# 1. The script that we want to use to train the model (i.e. `entry_point`). This is the heart of the Script Mode method. Additionally, set the `script_mode` parameter to `True`.
# 1. The role which allows us access to the S3 bucket containing the train and test data set (i.e. `role`)
# 1. How many instances we want to use in training (i.e. `instance_count`) and what type of instance we want to use in training (i.e. `instance_type`)
# 1. Which version of scikit-learn to use (i.e. `framework_version`)
# 1. Training hyperparameters (i.e. `hyperparameters`)
# 
# After setting these parameters, the `fit` function is invoked to train the model.

# In[26]:


# Docs: https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html

from sagemaker.sklearn import SKLearn

sk_estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.c5.xlarge",
    py_version="py3",
    framework_version="1.2-1",
    script_mode=True,
    hyperparameters={"estimators": 20},
)

# Train the estimator
sk_estimator.fit({"train": training_input_path})


# ## Deploy and test endpoint
# After training the model, it is time to deploy it as an endpoint. To do so, we invoke the `deploy` function within the scikit-learn estimator. As shown in the code below, one can define the number of instances (i.e. `initial_instance_count`) and instance type (i.e. `instance_type`) used to deploy the model.

# In[ ]:


import time

sk_endpoint_name = "sklearn-rf-model" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
sk_predictor = sk_estimator.deploy(
    initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=sk_endpoint_name
)


# After the endpoint has been completely deployed, it can be invoked using the [SageMaker Runtime Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html) (which is the method used in the code cell below) or [Scikit Learn Predictor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-predictor). If you plan to use the latter method, make sure to use a [Serializer](https://sagemaker.readthedocs.io/en/stable/api/inference/serializers.html) to serialize your data properly.

# In[ ]:


import json

client = sagemaker_session.sagemaker_runtime_client

request_body = {"Input": [[9.0, 3571, 1976, 0.525]]}
data = json.loads(json.dumps(request_body))
payload = json.dumps(data)

response = client.invoke_endpoint(
    EndpointName=sk_endpoint_name, ContentType="application/json", Body=payload
)

result = json.loads(response["Body"].read().decode())["Output"]
print("Predicted class category {} ({})".format(result, categories_map[result]))


# ## Cleanup
# If the model and endpoint are no longer in use, they should be deleted to save costs and free up resources.

# In[ ]:


sk_predictor.delete_model()
sk_predictor.delete_endpoint()


# ## Notebook CI Test Results
# 
# This notebook was tested in multiple regions. The test results are as follows, except for us-west-2 which is shown at the top of the notebook.
# 
# ![This us-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This us-east-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-2/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This us-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This ca-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ca-central-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This sa-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/sa-east-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This eu-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This eu-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-2/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This eu-west-3 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-3/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This eu-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-central-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This eu-north-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-north-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This ap-southeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This ap-southeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-2/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This ap-northeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This ap-northeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-2/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
# ![This ap-south-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-south-1/sagemaker-script-mode|sklearn|sklearn_byom.ipynb)
# 
