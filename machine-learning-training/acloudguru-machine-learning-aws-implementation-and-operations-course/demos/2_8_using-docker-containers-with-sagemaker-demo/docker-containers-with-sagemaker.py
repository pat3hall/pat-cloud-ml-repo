#!/usr/bin/env python
# coding: utf-8

# # Using Docker Containers with Amazon SageMaker Demo
# 

# ---
# 
# Use this notebook with a SageMaker notebook Jupyter Lab, not using SageMaker Studio.
# 
# ---

# SageMaker, enables you to package your own algorithms that can than be trained and deployed in the SageMaker environment. 
# 
# This demo that shows how to build a Docker container for SageMaker and use it for training and inference, if there is no pre-built container matching your requirements that you can use.

# ## Part 1: Packaging and Uploading your Algorithm for use with Amazon SageMaker

# ### The parts of the sample container
# 
# In the `container` directory are all the components you need to package the sample algorithm for SageMager:
# 
#     .
#     |-- Dockerfile
#     |-- build_and_push.sh
#     `-- decision_trees
#         |-- nginx.conf
#         |-- predictor.py
#         |-- serve
#         |-- train
#         `-- wsgi.py
# 
# * __`Dockerfile`__ describes how to build your Docker container image. More details below.
# * __`build_and_push.sh`__ is a script that uses the Dockerfile to build your container images and then pushes it to ECR.
# * __`decision_trees`__ is the directory containing the files that will be installed in the container.
# * __`local_test`__ is a directory that shows how to test your new container on any computer that can run Docker. 
# 
# The files that we'll put in the container are:
# 
# * __`nginx.conf`__ is the configuration file for the nginx front-end. Generally, you should be able to take this file as-is.
# * __`predictor.py`__ is the program that actually implements the Flask web server and the decision tree predictions for this app. 
# * __`serve`__ is the program started when the container is started for hosting. It simply launches the gunicorn server which runs multiple instances of the Flask app defined in `predictor.py`. 
# * __`train`__ is the program that is invoked when the container is run for training. You can modify this program to implement your training algorithm.
# * __`wsgi.py`__ is a small wrapper used to invoke the Flask app. 

# ### The Dockerfile
# 
# Docker uses a simple file called a `Dockerfile` to specify how the image is assembled. 
# 
# SageMaker uses Docker to allow users to train and deploy models, inculding creating your own.
# 
# The Dockerfile describes the image that we want to build. You can think of it as describing the complete operating system installation of the system that you want to run. A Docker container running is quite a bit lighter than a full operating system, however, because it takes advantage of Linux on the host machine for the basic operations. 
# 
# We'll use a standard Ubuntu installation and install the things needed by our model. 
# Then add the code that implements our specific algorithm to the container and set up the right environment to run under.
# 
# Let's review the Dockerfile:

# In[1]:


get_ipython().system('cat container/Dockerfile')


# ### Building and registering the container
# 
# Build the container image using `docker build`. 
# Push the container image to ECR using `docker push`. 
# 
# This code looks for an ECR repository in your account. If the repository doesn't exist, the script will create it.

# In[ ]:


get_ipython().run_cell_magic('sh', '', '\n# The name of our algorithm\nalgorithm_name=sagemaker-decision-trees\n\ncd container\n\nchmod +x decision_trees/train\nchmod +x decision_trees/serve\n\naccount=$(aws sts get-caller-identity --query Account --output text)\n\n# Get the region defined in the current configuration (default to us-east-1 if none defined)\nregion=$(aws configure get region)\nregion=${region:-us-east-1}\n\nfullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"\n\n# If the repository doesn\'t exist in ECR, create it.\naws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1\n\nif [ $? -ne 0 ]\nthen\n    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null\nfi\n\n# Get the login command from ECR and execute it directly\naws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}\n\n# Build the docker image locally with the image name and then push it to ECR\n# with the full name.\n\ndocker build -t ${algorithm_name} .\ndocker tag ${algorithm_name} ${fullname}\n\ndocker push ${fullname}\n')


# ## Part 2: Using your Algorithm in Amazon SageMaker
# 
# Once you have your container packaged, you can use it to train models and use the model for hosting or batch transforms. Let's do that with the algorithm we made above.
# 
# ## Set up the environment
# 
# Here we specify a bucket to use and the role that will be used for working with SageMaker.

# In[ ]:


# S3 prefix
prefix = "DEMO-scikit-byo-iris"

# Define IAM role
import boto3
import re

import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role

role = get_execution_role()


# ## Create the session
# 
# The session remembers our connection parameters to SageMaker. We'll use it to perform all of our SageMaker operations.

# In[ ]:


import sagemaker as sage
from time import gmtime, strftime

sess = sage.Session()


# ## Upload the data for training
# 
# For the purposes of this example, we're using some the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), which is in the training folder. 

# In[ ]:


WORK_DIRECTORY = "data"

data_location = sess.upload_data(WORK_DIRECTORY, key_prefix=prefix)


# ## Train the Model
# 
# In order to use SageMaker to train our algorithm, we'll create an `Estimator` that defines how to use the container to train. This includes the configuration we need to invoke SageMaker training:
# 
# * The __container name__. This is defined above.
# * The __role__. As defined above.
# * The __instance count__ The number of EC2 instances to use for training.
# * The __instance type__ Type of EC2 instance to use for training.
# * The __output path__ Where the model artifact will be written.
# * The __session__ is the SageMaker session object defined above.
# 
# Then we use fit() on the estimator to train against the data that we uploaded above.

# In[ ]:


account = sess.boto_session.client("sts").get_caller_identity()["Account"]
region = sess.boto_session.region_name
image = "{}.dkr.ecr.{}.amazonaws.com/sagemaker-decision-trees:latest".format(account, region)

tree = sage.estimator.Estimator(
    image,
    role,
    1,
    "ml.m5.large",
    output_path="s3://{}/output".format(sess.default_bucket()),
    sagemaker_session=sess,
)

tree.fit(data_location)


# ## Deploying the model
# 
# After training is complete, deploy the model using the `deploy` API call. Provide the instance count, instance type, and optionally serializer and deserializer functions. 

# In[ ]:


from sagemaker.serializers import CSVSerializer

predictor = tree.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    serializer=CSVSerializer()
)


# ### Choose some data and use it for a prediction
# 
# Make sure the model deployed properly by running some predictions, we'll re-use some of the data we used for training, for the purpose of checking that the model successfully deployed. 
# 
# Choose some data and use it for a prediction
# In order to do some predictions, we'll extract some of the data we used for training and do predictions against it. This is, of course, bad statistical practice, but a good way to see how the mechanism works.

# In[ ]:


shape = pd.read_csv("data/iris.csv", header=None)
shape.sample(3)


# In[ ]:


# drop the label column in the training set
shape.drop(shape.columns[[0]], axis=1, inplace=True)
shape.sample(3)


# In[ ]:


import itertools

a = [50 * i for i in range(3)]
b = [40 + i for i in range(10)]
indices = [i + j for i, j in itertools.product(a, b)]

test_data = shape.iloc[indices[:-1]]


# Prediction is as easy as calling predict with the predictor we got back from deploy and the data we want to do predictions with. The serializers take care of doing the data conversions for us.

# In[ ]:


print(predictor.predict(test_data.values).decode("utf-8"))


# ### Optional cleanup
# When you're done with the endpoint, you'll want to clean it up.

# In[ ]:


sage.Session().delete_endpoint(predictor.endpoint)

