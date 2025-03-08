#!/usr/bin/env python
# coding: utf-8

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
from sagemaker.amazon.amazon_estimator import get_image_uri


# In[4]:


boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# In[5]:


sess = sagemaker.Session()
region = sess.boto_region_name
print(region)


# In[6]:


# role_arn = sagemaker.get_execution_role()
role_arn = os.getenv("SGMKR_ROLE_ARN")


# In[7]:


bucket_name = "pat-demo-bkt"
data_path = "sgmkr_clf_lst"

nclasses = 3
nimgs_train = 36
nepochs = 10
mini_batch_size = 8

train_instance_type = "ml.g4dn.xlarge"
job_name_prefix = "flowers-clf-ib-resent-"


# https://aws.amazon.com/sagemaker/pricing/

# In[ ]:





# In[8]:


train_image_uri = sagemaker.image_uris.retrieve(
    framework="image-classification",
    region=region,
    image_scope="training",
    version="latest",
)
print(train_image_uri)


# In[9]:


s3_output_path = "s3://{}/{}/{}".format(bucket_name, data_path, "model_output")


# In[10]:


clf_estimator = sagemaker.estimator.Estimator(
    image_uri=train_image_uri,
    role=role_arn,
    instance_count=1,
    instance_type=train_instance_type,
    volume_size=50,
    max_run=360000,
    input_mode="File",
    output_path=s3_output_path,
    sagemaker_session=sess,
)


# In[ ]:





# In[11]:


clf_estimator.set_hyperparameters(
    num_classes=nclasses,  # update this
    epochs=nepochs,  # update this
    num_training_samples=nimgs_train,  # update this
    mini_batch_size=mini_batch_size,  # update this
    num_layers=18,
    use_pretrained_model=1,
    image_shape="3,224,224",
    resize=256,
    learning_rate=0.001,
    use_weighted_loss=1,
    augmentation_type="crop_color_transform",
    precision_dtype="float32",
    multi_label=0,
)


# In[12]:


s3_train_imgs = "s3://{}/{}/{}".format(bucket_name, data_path, "train_imgs")
s3_valid_imgs = "s3://{}/{}/{}".format(bucket_name, data_path, "valid_imgs")
s3_train_annot = "s3://{}/{}/{}".format(bucket_name, data_path, "train_annots")
s3_valid_annot = "s3://{}/{}/{}".format(bucket_name, data_path, "valid_annots")

train_imgs = sagemaker.inputs.TrainingInput(
    s3_train_imgs,
    distribution="FullyReplicated",
    content_type="application/jpeg",
    s3_data_type="S3Prefix",
)
valid_imgs = sagemaker.inputs.TrainingInput(
    s3_valid_imgs,
    distribution="FullyReplicated",
    content_type="application/jpeg",
    s3_data_type="S3Prefix",
)
train_annot = sagemaker.inputs.TrainingInput(
    s3_train_annot,
    distribution="FullyReplicated",
    content_type="application/jpeg",
    s3_data_type="S3Prefix",
)
valid_annot = sagemaker.inputs.TrainingInput(
    s3_valid_annot,
    distribution="FullyReplicated",
    content_type="application/jpeg",
    s3_data_type="S3Prefix",
)

data_channels = {
    "train": train_imgs,
    "validation": valid_imgs,
    "train_lst": train_annot,
    "validation_lst": valid_annot,
}


# In[13]:


timestamp = (
    str(datetime.now().replace(microsecond=0)).replace(" ", "-").replace(":", "-")
)
job_name = job_name_prefix + timestamp
print(job_name)


# In[ ]:





# In[15]:


clf_estimator.fit(inputs=data_channels, logs=True, job_name=job_name)


# In[ ]:





# In[16]:


infer_instance_type = "ml.t2.medium"
model_name = job_name
endpoint_name = job_name


# In[19]:


clf_predictor = clf_estimator.deploy(
    initial_instance_count=1,
    instance_type=infer_instance_type,
    endpoint_name=endpoint_name,
    model_name=model_name,
)


# In[20]:


sgmkr_runt = boto3.client("runtime.sagemaker")


# In[21]:


with open("images/rose.jpg", "rb") as image:
        payload = image.read()
        payload = bytearray(payload)

response = sgmkr_runt.invoke_endpoint(
    EndpointName = endpoint_name,
    ContentType = 'image/jpeg',
    Accept = "application/json;verbose",
    Body = payload,
)

prediction = json.loads(response['Body'].read().decode())
print(prediction)


# In[22]:


clf_predictor.delete_endpoint()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




