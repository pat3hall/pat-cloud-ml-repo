#!/usr/bin/env python
# coding: utf-8

# ## Image Classification - TensorFlow

# https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/image_classification_tensorflow/Amazon_TensorFlow_Image_Classification.ipynb
#
#
# https://github.com/aws/amazon-sagemaker-examples/blob/93fc48d21bf88d07853775f11d6ef7db92110549/introduction_to_amazon_algorithms/jumpstart_image_classification/Amazon_JumpStart_Image_Classification.ipynb
#
#
# https://aws.amazon.com/blogs/machine-learning/transfer-learning-for-tensorflow-image-classification-models-in-amazon-sagemaker/
#
#
# https://aws.amazon.com/blogs/machine-learning/run-image-classification-with-amazon-sagemaker-jumpstart/

# In[1]:


#%load_ext nb_black


# In[2]:


import os
import json
import logging
from datetime import datetime


# In[3]:


import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker import image_uris,  script_uris
from sagemaker import hyperparameters
from sagemaker.estimator import Estimator


# In[4]:


boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# In[5]:


sess = sagemaker.Session()
region = sess.boto_region_name
print(region)


# In[7]:


# role_arn = sagemaker.get_execution_role()
role_arn = os.getenv("SGMKR_ROLE_ARN")


# In[8]:


bucket_name = "pat-demo-bkt"
data_path = "sgmkr_clf_subfolders"

nepochs = 10
mini_batch_size = 8

train_instance_type = "ml.g4dn.xlarge"
job_name_prefix = "flowers-clf-js-tf-"


# In[ ]:





# In[9]:


model_id = "pytorch-ic-mobilenet-v2"
model_version = "*"


# In[9]:


train_image_uri = image_uris.retrieve(
    region=None,
    framework=None,
    model_id=model_id,
    model_version=model_version,
    image_scope="training",
    instance_type=train_instance_type,
)

train_source_uri = script_uris.retrieve(
    model_id=model_id, model_version=model_version, script_scope="training"
)

train_model_uri = model_uris.retrieve(
    model_id=model_id, model_version=model_version, model_scope="training"
)

print(train_image_uri)
print(train_source_uri)
print(train_model_uri)


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


hyperparameters = hyperparameters.retrieve_default(
    model_id=model_id, model_version=model_version
)

hyperparameters["epochs"] = "5"
print(hyperparameters)


# In[11]:


s3_output_path = "s3://{}/{}/{}".format(bucket_name, data_path, "model_output")


# In[12]:


clf_estimator = Estimator(
    role=role_arn,
    image_uri=train_image_uri,
    source_dir=train_source_uri,
    model_uri=train_model_uri,
    entry_point="transfer_learning.py",
    instance_count=1,
    instance_type=train_instance_type,
    max_run=360000,
    hyperparameters=hyperparameters,
    output_path=s3_output_path,
)


# In[ ]:





# In[ ]:





# In[17]:


s3_train_imgs = "s3://{}/{}/{}".format(bucket_name, data_path, "train_imgs")
s3_valid_imgs = "s3://{}/{}/{}".format(bucket_name, data_path, "valid_imgs")
data_channels = {
    "training": s3_train_imgs,
    "validation": s3_valid_imgs,
}
print(data_channels)


# In[ ]:





# In[ ]:





# In[18]:


timestamp = (
    str(datetime.now().replace(microsecond=0)).replace(" ", "-").replace(":", "-")
)
job_name = job_name_prefix + timestamp
print(job_name)


# In[19]:


clf_estimator.fit(inputs=data_channels, logs=True, job_name=job_name)


# In[ ]:





# In[ ]:





# In[20]:


infer_instance_type = "ml.t2.medium"


# In[21]:


deploy_image_uri = image_uris.retrieve(
    region=None,
    framework=None,
    image_scope="inference",
    model_id=model_id,
    model_version=model_version,
    instance_type=infer_instance_type,
)

deploy_source_uri = script_uris.retrieve(
    model_id=model_id, model_version=model_version, script_scope="inference"
)


# In[22]:


model_name = job_name
endpoint_name = job_name


# In[24]:


clf_predictor = clf_estimator.deploy(
    initial_instance_count=1,
    instance_type=infer_instance_type,
    entry_point="inference.py",
    image_uri=deploy_image_uri,
    source_dir=deploy_source_uri,
    endpoint_name=endpoint_name,
    model_name=model_name,
)


# In[25]:


sgmkr_runt = boto3.client("runtime.sagemaker")


# In[26]:


with open("images/rose.jpg", "rb") as image:
    payload = image.read()
    # payload = bytearray(payload)

response = sgmkr_runt.invoke_endpoint(
    EndpointName=endpoint_name,
    # ContentType = 'image/jpeg',
    ContentType="application/x-image",
    Accept="application/json;verbose",
    Body=payload,
)

prediction = json.loads(response["Body"].read().decode())
print(prediction)


# In[27]:


clf_predictor.delete_endpoint()


# In[ ]:





# In[ ]:




