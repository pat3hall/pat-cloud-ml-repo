#!/usr/bin/env python
# coding: utf-8

# ## Object Detection - Inbuilt SSD model

# In[1]:


from IPython.display import Image
Image("https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/627d124572023b6948b6cdff_60ed9a4e09e2c648f1b8a013_object-detection-cover.png")


# In[2]:


#%load_ext nb_black


# In[3]:


import os
import json
import logging
from datetime import datetime


# In[4]:


import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri


# In[5]:


boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# In[6]:


sess = sagemaker.Session()
region = sess.boto_region_name
print(region)


# In[7]:


# role_arn = sagemaker.get_execution_role()
role_arn = os.getenv("SGMKR_ROLE_ARN")


# In[8]:


bucket_name = "pat-demo-bkt"
data_path = "sgmkr_od_ssd"

nclasses = 3
nimgs_train = 36
nepochs = 10
mini_batch_size = 8

train_instance_type = "ml.g4dn.xlarge"
job_name_prefix = "flowers-od-ib-ssd-"


# In[ ]:





# In[ ]:





# In[8]:


train_image_uri = sagemaker.image_uris.retrieve(
    framework="object-detection",
    region=region,
    image_scope="training",
    version="latest",
)


# In[ ]:





# In[9]:


s3_output_path = "s3://{}/{}/{}".format(bucket_name, data_path, "model_output")


# In[10]:


od_estimator = sagemaker.estimator.Estimator(
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


# In[11]:


od_estimator.set_hyperparameters(
    num_classes=nclasses,  # update this
    num_training_samples=nimgs_train,  # update this
    epochs=nepochs,  # update this
    mini_batch_size=mini_batch_size,  # update this
    base_network="resnet-50",  # Transfer Learning
    use_pretrained_model=1,  # IMP
    learning_rate=0.002,
    lr_scheduler_step="10",
    lr_scheduler_factor=0.1,
    optimizer="sgd",
    momentum=0.9,
    weight_decay=0.0005,
    overlap_threshold=0.5,
    nms_threshold=0.45,
    image_shape=512,
    label_width=50,
)


# In[12]:


s3_train_imgs = "s3://{}/{}/{}".format(bucket_name, data_path, "train_imgs")
s3_valid_imgs = "s3://{}/{}/{}".format(bucket_name, data_path, "valid_imgs")
s3_train_annot = "s3://{}/{}/{}".format(bucket_name, data_path, "train_annots")
s3_valid_annot = "s3://{}/{}/{}".format(bucket_name, data_path, "valid_annots")

train_imgs = sagemaker.inputs.TrainingInput(
    s3_train_imgs,
    distribution="FullyReplicated",
    content_type="image/jpeg",
    s3_data_type="S3Prefix",
)
valid_imgs = sagemaker.inputs.TrainingInput(
    s3_valid_imgs,
    distribution="FullyReplicated",
    content_type="image/jpeg",
    s3_data_type="S3Prefix",
)
train_annot = sagemaker.inputs.TrainingInput(
    s3_train_annot,
    distribution="FullyReplicated",
    content_type="image/jpeg",
    s3_data_type="S3Prefix",
)
valid_annot = sagemaker.inputs.TrainingInput(
    s3_valid_annot,
    distribution="FullyReplicated",
    content_type="image/jpeg",
    s3_data_type="S3Prefix",
)

data_channels = {
    "train": train_imgs,
    "validation": valid_imgs,
    "train_annotation": train_annot,
    "validation_annotation": valid_annot,
}


# In[13]:


timestamp = (
    str(datetime.now().replace(microsecond=0)).replace(" ", "-").replace(":", "-")
)
job_name = job_name_prefix + timestamp
print(job_name)


# In[14]:


od_estimator.fit(inputs=data_channels, logs=True, job_name=job_name)


# In[15]:


infer_instance_type = "ml.t2.medium"
model_name = job_name
endpoint_name = job_name


# In[18]:


od_predictor = od_estimator.deploy(
    initial_instance_count=1,
    instance_type=infer_instance_type,
    endpoint_name=endpoint_name,
    model_name=model_name,
)


# In[ ]:





# In[ ]:





# In[19]:


sgmkr_runt = boto3.client("runtime.sagemaker")


# In[23]:


with open("images/rose.jpg", "rb") as image:
        payload = image.read()
        payload = bytearray(payload)

response = sgmkr_runt.invoke_endpoint(
    EndpointName = endpoint_name,
    ContentType = 'image/jpeg',
    #Accept = "application/json;n_predictions=5",
    Body = payload,
)

prediction = json.loads(response['Body'].read().decode())
print(prediction)


# In[24]:


od_predictor.delete_endpoint()


# In[ ]:





# In[ ]:





# In[ ]:




