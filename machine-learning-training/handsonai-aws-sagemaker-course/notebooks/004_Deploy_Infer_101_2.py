#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%load_ext nb_black


# In[2]:


import os
import boto3
import logging
from datetime import datetime

boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# In[3]:


import sagemaker
from sagemaker.transformer import Transformer


# In[6]:


sgmkr_clnt = boto3.client("sagemaker")
sgmkr_rt = boto3.client("runtime.sagemaker")


# In[7]:


# role_arn = sagemaker.get_execution_role()
role_arn = os.getenv("SGMKR_ROLE_ARN")


# #### Create model

# In[8]:


from sagemaker import image_uris


# In[12]:


region = sagemaker.Session().boto_region_name
print(region)
model_img = sagemaker.image_uris.retrieve("xgboost", region, "latest")
print(model_img)


# In[13]:


#model_img = "811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest"


# In[23]:


get_ipython().system('aws s3 ls pat-demo-bkt')


# In[24]:


bucket = 'pat-demo-bkt'
prefix = 'iris/model/iris-xgboost-2024-10-17-17-03-48/output'
model_tar_file = "model.tar.gz"
s3_model_path = "s3://{}/{}/{}".format(bucket, prefix, model_tar_file)
s3_model_path_ls = "{}/{}/{}".format(bucket, prefix, model_tar_file)
#s3_model_path = "s3://pat-demo-bkt/iris/model/iris-xgboost-2024-10-17-17-03-48/output/model.tar.gz"
print (s3_model_path_ls)
get_ipython().system('aws s3 ls {s3_model_path_ls}')


# In[25]:


model_path = (
    s3_model_path
)
print (model_path)


# In[26]:


model_name = "iris-xgboost-" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
print(model_name)


# In[27]:


response = sgmkr_clnt.create_model(
    ModelName=model_name,
    PrimaryContainer={"Image": model_img, "ModelDataUrl": model_path},
    ExecutionRoleArn=role_arn,
)

print(response)


# In[28]:


bucket = "pat-demo-bkt"
prefix_bt = "iris"


# #### Batch Transform

# In[29]:


batch_ip = 's3://{}/{}/{}'.format(bucket, prefix_bt, 'batch_transform')
batch_op = 's3://{}/{}/{}'.format(bucket, prefix_bt, 'batch_transform')


# In[30]:


transformer = Transformer(
    model_name=model_name,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=batch_op,
)

transformer.transform(
    data=batch_ip, data_type="S3Prefix", content_type="text/csv"
)
transformer.wait()


# #### Endpoint

# In[31]:


ep_config_name = "tmp-ep-config-" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
print(ep_config_name)


# In[32]:


response = sgmkr_clnt.create_endpoint_config(
    EndpointConfigName=ep_config_name,
    ProductionVariants=[
        {
            "VariantName": "version-1",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": "ml.m4.xlarge",
            # sever_less = ''
        },
    ],
)

print(response)


# In[33]:


ep_name = "tmp-ep-" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
print(ep_name)


# In[34]:


response = sgmkr_clnt.create_endpoint(
    EndpointName=ep_name, EndpointConfigName=ep_config_name,
)
print(response)


# In[35]:


waiter = sgmkr_clnt.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=ep_name, WaiterConfig={"Delay": 123, "MaxAttempts": 123})
print("Endpoint created")


# In[36]:


payload = "7.7, 3.0, 6.1, 2.3"
# payload = '7.7, 3.0, 6.1, 2.3 \n 7.9, 3.8, 6.4, 2.1'


# In[37]:


sgmkr_runt = boto3.client("runtime.sagemaker")


# In[38]:


response = sgmkr_runt.invoke_endpoint(
    EndpointName=ep_name, ContentType="text/csv", Body=payload,
)

prediction = response["Body"].read().decode()
print(prediction)


# In[40]:


print("Deleting sagemaker endpoint")
response = sgmkr_clnt.delete_endpoint(EndpointName = ep_name)
print("Deleted sagemaker endpoint")
#print(response)


# In[41]:


print("Deleting sagemaker endpoint configuration")
response = sgmkr_clnt.delete_endpoint_config(EndpointConfigName = ep_config_name)
print("Deleted sagemaker endpoint configuration")
#print(response)


# In[ ]:




