#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install -q -U sagemaker


# In[2]:


#%load_ext nb_black


# In[3]:


import os
import logging
from datetime import datetime


# In[5]:


import boto3
import sagemaker
from sagemaker.session import TrainingInput
from sagemaker import image_uris
from sagemaker import hyperparameters


# In[6]:


boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# In[7]:


region = sagemaker.Session().boto_region_name
print(region)


# In[6]:


# role_arn = sagemaker.get_execution_role()
role_arn = os.getenv("SGMKR_ROLE_ARN")


# In[7]:


bucket = "pat-demo-bkt"
prefix = "iris"


# In[8]:


get_ipython().system('aws s3 ls {bucket}/{prefix}/')


# In[9]:


get_ipython().system('aws s3 ls {bucket}/{prefix}/data/ --recursive')


# In[10]:


train_file = "data/iris_train.csv"
valid_file = "data/iris_test.csv"

train_file_uri = "s3://{}/{}/{}".format(bucket, prefix, train_file)
valid_file_uri = "s3://{}/{}/{}".format(bucket, prefix, valid_file)
print("train file uri:", train_file_uri)
print("valid file uri:", valid_file_uri)


# In[11]:


train_ip = TrainingInput(train_file_uri, content_type="csv")
print(train_ip)


# In[12]:


valid_ip = TrainingInput(valid_file_uri, content_type="csv")
print(valid_ip)


# In[13]:


model_op = "s3://{}/{}/{}".format(bucket, prefix, "model")
print(model_op)


# In[14]:


model_img = sagemaker.image_uris.retrieve("xgboost", region, "latest")
print(model_img)


# In[15]:


base_job_name = "iris-xgboost-"


# In[16]:


xgb_model = sagemaker.estimator.Estimator(
    image_uri=model_img,
    role=role_arn,
    base_job_name=base_job_name,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path=model_op,
    sagemaker_session=sagemaker.Session(),
    volume_size=5,
)


# In[17]:


xgb_model.set_hyperparameters(
    num_class=3, max_depth=5, num_round=10, objective="multi:softmax",
)


# In[18]:


job_name = base_job_name + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
print(job_name)


# In[19]:


xgb_model.fit({"train": train_ip, "validation": valid_ip}, wait=True, job_name=job_name)


# In[20]:


get_ipython().system('aws s3 ls {bucket}/{prefix}/model/')


# In[21]:


get_ipython().system('aws s3 ls {bucket}/{prefix}/model/{job_name}/')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




