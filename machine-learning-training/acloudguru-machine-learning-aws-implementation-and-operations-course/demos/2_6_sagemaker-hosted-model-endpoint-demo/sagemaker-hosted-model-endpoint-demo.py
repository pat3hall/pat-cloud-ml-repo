#!/usr/bin/env python
# coding: utf-8

# # Running Real-Time Predictions using a SageMaker Hosted Model Endpoint 
# Using Linear Learner with the MNIST dataset to predict whether a hand writen digit is a 0 or not. 
# Based on the following AWS sample: https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist.ipynb
# 

# ## Introduction
# 
# The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset consists of images of handwritten digits, from zero to nine.  The individual pixel values from each 28 x 28 grayscale image of the digit will be used to predict a yes or no label of whether the digit is a 0 or some other digit (1, 2, 3, ... 9).
# 
# Linear Learner will be used to perform a binary classification. The `predicted_label` will take a value of either `0` or `1` where `1` denotes that we predict the image is a 0, while `0` denotes that we are predicting the image is not of a 0.

# ## Prequisites and Preprocessing
# 
# The notebook works with SageMaker Studio Jupyter Lab.
# 
# Specify:
# 
# - The S3 bucket and prefix to use for training and model data. 
# - The IAM role arn used to give training and hosting access to your data

# In[1]:


import sagemaker

bucket = sagemaker.Session().default_bucket()
prefix = "sagemaker/DEMO-linear-mnist"

# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# ### Data ingestion
# 
# Ingest the dataset from an online URL into memory, for preprocessing prior to training. As it's a small data set, we can do this in memory.

# In[2]:


get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\nfobj = (\n    boto3.client("s3")\n    .get_object(\n        Bucket=f"sagemaker-example-files-prod-{boto3.session.Session().region_name}",\n        Key="datasets/image/MNIST/mnist.pkl.gz",\n    )["Body"]\n    .read()\n)\n\nwith open("mnist.pkl.gz", "wb") as f:\n    f.write(fobj)\n\n# Load the dataset\nwith gzip.open("mnist.pkl.gz", "rb") as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")\n')


# ### Data inspection
# 
# Once the dataset is imported we can inspect at one of the digits that is part of the dataset.

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (2, 10)


def show_digit(img, caption="", subplot=None):
    if subplot == None:
        _, (subplot) = plt.subplots(1, 1)
    imgr = img.reshape((28, 28))
    subplot.axis("off")
    subplot.imshow(imgr, cmap="gray")
    plt.title(caption)


show_digit(train_set[0][30], "This is a {}".format(train_set[1][30]))


# ### Convert the Data to recordIO-wrapped protobuf format
# 
# Amazon SageMaker's version of Linear Learner takes recordIO-wrapped protobuf (or CSV) So we need to convert the data to a suppported format so the algorithm can use it.
# 
# The following code converts the np.array to recordIO-wrapped protobuf format.

# In[4]:


import io
import numpy as np
import sagemaker.amazon.common as smac

vectors = np.array([t.tolist() for t in train_set[0]]).astype("float32")
labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype("float32")

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)


# ## Upload training data
# Now that we've created our recordIO-wrapped protobuf, we'll need to upload it to S3, so that Amazon SageMaker training can use it.

# In[5]:


import boto3
import os

key = "recordio-pb-data"
boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, "train", key)).upload_fileobj(buf)
s3_train_data = "s3://{}/{}/train/{}".format(bucket, prefix, key)
print("uploaded training data location: {}".format(s3_train_data))


# Setup an output S3 location for the model artifact that will be output as the result of training with the algorithm.

# In[6]:


output_location = "s3://{}/{}/output".format(bucket, prefix)
print("training artifacts will be uploaded to: {}".format(output_location))

## Training the linear model

Train the model, and monitor status until it is completed.  In this example that takes between 7 and 11 minutes.  

First, specify the container, we're using the linear learner framework.
# In[7]:


from sagemaker.image_uris import retrieve

container = retrieve("linear-learner", boto3.Session().region_name)


# Start the training job. 
# - `feature_dim` is 784, which is the number of pixels in each 28 x 28 image.
# - `predictor_type` is 'binary_classifier' - we are trying to predict whether the image is or is not a 0.
# - `mini_batch_size` is set to 200.  

# In[8]:


import boto3

sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(
    container,
    role,
    train_instance_count=1,
    train_instance_type="ml.m5.large",
    output_path=output_location,
    sagemaker_session=sess,
)
linear.set_hyperparameters(feature_dim=784, predictor_type="binary_classifier", mini_batch_size=200)

linear.fit({"train": s3_train_data})


# ## Configure a Model Endpoint
# After training is completed, we can deploy our model using a SageMaker real-time hosted endpoint. 
# This will allow us to make predictions (or inference) from the model dynamically.
# 
# Note we are using the deploy API call, specifying the number of initial instances, and instance type, also specify how to serialize requests and deserialize responses, so the input will be our data in recordIO-wrapped protobuf format, output is going to be in JSON format. 

# In[9]:


from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

linear_predictor = linear.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer(),
)


# ## Validate the model for use
# Finally, we can now validate the model for use.  We can pass HTTP POST requests to the endpoint to get back predictions.  To make this easier, we'll again use the Amazon SageMaker Python SDK and specify how to serialize requests and deserialize responses that are specific to the algorithm.

# Now let's try getting a prediction for a single record.

# In[10]:


result = linear_predictor.predict(train_set[0][30:31])
print(result)


# If everything works, the endpoint will return a prediction: `predicted_label` which will be either `0` or `1`. `1` denotes that we predict the image is a 0, while `0` denotes that we are predicting the image is not of a 0.
# 
# It also gives a `score` which is a single floating point number indicating how strongly the algorithm believes it has predicted correctly. 

# ### Clean up - Delete the Endpoint
# 
# The delete_endpoint line in the cell below will remove the hosted endpoint to avoid any unnecessary charges.
# We should also delete the S3 buckets as well. 

# In[11]:


sagemaker.Session().delete_endpoint(linear_predictor.endpoint)


# In[ ]:




