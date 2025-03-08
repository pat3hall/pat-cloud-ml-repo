#!/usr/bin/env python
# coding: utf-8

# ## Object Detection - TensorFlow SSD model

# In[1]:


from IPython.display import Image
Image("https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/627d124572023b6948b6cdff_60ed9a4e09e2c648f1b8a013_object-detection-cover.png")


# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# In[32]:


import os
import json
import logging
from datetime import datetime


# In[14]:


import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.estimator import Estimator
from sagemaker import hyperparameters


# In[4]:


boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# In[5]:


sess = sagemaker.Session()
region = sess.boto_region_name
print(region)


# In[6]:


# role_arn = sagemaker.get_execution_role()
role_arn = os.getenv("SGMKR_ROLE_ARN")


# In[20]:


bucket_name = "pat-demo-bkt"
data_path = "sgmkr_od_tf"

# nclasses = 3
# nimgs_train = 36
nepochs = 10
mini_batch_size = 8

train_instance_type = "ml.g4dn.xlarge"
job_name_prefix = "flowers-od-js-tf-"


# In[ ]:





# In[11]:


model_id = "tensorflow-od1-ssd-resnet50-v1-fpn-640x640-coco17-tpu-8"
model_version = "*"
train_instance_type = "ml.g4dn.xlarge"


# In[12]:


train_image_uri = image_uris.retrieve(
    model_id=model_id,
    model_version=model_version,
    image_scope="training",
    instance_type=train_instance_type,
    region=None,
    framework=None,
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


# In[15]:


s3_output_path = "s3://{}/{}/{}".format(bucket_name, data_path, "model_output")


# In[ ]:





# In[16]:


hyperparameters = hyperparameters.retrieve_default(
    model_id=model_id,
    model_version=model_version
)
print(hyperparameters)
hyperparameters["epochs"] = "5"
hyperparameters['train_only_top_layer'] = True


# In[17]:


tf_od_estimator = Estimator(
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





# In[18]:


s3_train_imgs_annot = "s3://{}/{}/{}".format(bucket_name, data_path, "train/")
s3_valid_imgs_annot = "s3://{}/{}/{}".format(bucket_name, data_path, "valid/")

data_channels = {
    "training": s3_train_imgs_annot,
    # "validation": s3_valid_imgs_annot,
}
print(data_channels)


# In[19]:


timestamp = (
    str(datetime.now().replace(microsecond=0)).replace(" ", "-").replace(":", "-")
)
job_name = job_name_prefix + timestamp
print(job_name)


# In[22]:


tf_od_estimator.fit(inputs=data_channels, logs=True, job_name=job_name)


# In[23]:





# In[25]:


infer_instance_type = "ml.t2.medium"


# In[26]:


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


# In[27]:


model_name = job_name
endpoint_name = job_name


# In[28]:


od_predictor = tf_od_estimator.deploy(
    initial_instance_count=1,
    instance_type=infer_instance_type,
    entry_point="inference.py",
    image_uri=deploy_image_uri,
    source_dir=deploy_source_uri,
    endpoint_name=endpoint_name,
    model_name=model_name,
)


# In[29]:


sgmkr_runt = boto3.client("runtime.sagemaker")


# In[33]:


with open("images/rose.jpg", "rb") as image:
    payload = image.read()
    # payload = bytearray(payload)

response = sgmkr_runt.invoke_endpoint(
    EndpointName=endpoint_name,
    # ContentType = 'image/jpeg',
    ContentType="application/x-image",
    Accept="application/json;n_predictions=5",
    Body=payload,
)

prediction = json.loads(response["Body"].read().decode())
print(prediction)


# In[34]:


od_predictor.delete_endpoint()


# In[ ]:




