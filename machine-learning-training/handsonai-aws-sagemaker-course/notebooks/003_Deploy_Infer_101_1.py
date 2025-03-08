#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%load_ext nb_black


# In[2]:


import os
import io
from datetime import datetime
import logging


# In[3]:


import boto3
import sagemaker
from sagemaker.session import TrainingInput
from sagemaker import image_uris
from sagemaker import hyperparameters


# In[4]:


boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# In[5]:


region = sagemaker.Session().boto_region_name
print(region)


# In[7]:


# role_arn = sagemaker.get_execution_role()
role_arn = os.getenv("SGMKR_ROLE_ARN")


# In[8]:


bucket = "pat-demo-bkt"
prefix = "iris"


# In[9]:


get_ipython().system('aws s3 ls {bucket}/{prefix}/')


# In[10]:


get_ipython().system('aws s3 ls {bucket}/{prefix}/data/ --recursive')


# In[11]:


train_file = "data/iris_train.csv"
valid_file = "data/iris_test.csv"


# In[12]:


train_ip = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, train_file), content_type="csv"
)
valid_ip = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, valid_file), content_type="csv"
)


# In[13]:


model_op = "s3://{}/{}/{}".format(bucket, prefix, "model")


# In[14]:


train_image_uri = sagemaker.image_uris.retrieve("xgboost", region, "latest")
print(train_image_uri)


# In[38]:


base_job_name = "iris-xgboost"


# In[41]:


xgb_estimator = sagemaker.estimator.Estimator(
    image_uri=train_image_uri,
    role=role_arn,
    base_job_name=base_job_name,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    volume_size=5,
    output_path=model_op,
    sagemaker_session=sagemaker.Session(),
)


# In[42]:


xgb_estimator.set_hyperparameters(
    num_class=3, max_depth=5, num_round=10, objective="multi:softmax",
)


# In[43]:


# xgb_estimator.set_hyperparameters(
#     num_class=3,
#     max_depth=5,
#     eta=0.2,
#     gamma=4,
#     min_child_weight=6,
#     subsample=0.7,
#     objective="multi:softmax",
#     num_round=10,
# )


# In[47]:


job_name = "iris-xgboost-" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
print(job_name)


# In[48]:


xgb_estimator.fit(
    {"train": train_ip, "validation": valid_ip}, wait=True, job_name=job_name
)


# In[72]:


get_ipython().system('aws s3 ls {bucket}/{prefix}/model/')


# ### Inference

# In[50]:


from sagemaker.serializers import CSVSerializer


# #### Deploy the model as an endpoint

# In[51]:


type(xgb_estimator)


# In[52]:


xgb_predictor = xgb_estimator.deploy(
    initial_instance_count=1, instance_type="ml.t2.medium", serializer=CSVSerializer()
)


# #### Predictor single record

# In[53]:


xgb_predictor.predict("7.7, 3.0, 6.1, 2.3")


# #### Endpoint

# In[54]:


endpoint_name = xgb_predictor.endpoint_name
print(endpoint_name)


# In[55]:


sgmkr_runtime = boto3.client("runtime.sagemaker")


# In[ ]:





# #### Endpoint - One record

# In[56]:


payload_csv_text = "7.7, 3.0, 6.1, 2.3"
response = sgmkr_runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="text/csv", Body=payload_csv_text
)
print(response)


# In[57]:


print(response["Body"].read().decode())


# #### Endpoint - Multiple records

# In[58]:


payload_csv_text = "7.7, 3.0, 6.1, 2.3 \n 7.9, 3.8, 6.4, 2.1"

response = sgmkr_runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="text/csv", Body=payload_csv_text
)
print(response["Body"].read().decode())


# #### Endpoint - Multiple records from a local file

# In[59]:


csv_buffer = open("data/iris_infer.csv")
payload_csv_text = csv_buffer.read()

response = sgmkr_runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="text/csv", Body=payload_csv_text
)
print(response["Body"].read().decode())


# In[77]:


payload_csv_text


# #### Endpoint - Multiple records from a S3 file

# In[60]:


infer_ip_s3_uri = "s3://{}/{}/{}".format(
    bucket, prefix, "batch_transform/iris_infer.csv"
)

# payload_df = pd.read_csv(infer_ip_s3_uri)
# payload_df = wr.s3.read_csv(path=infer_ip_s3_uri)
s3_clnt = boto3.client("s3")
obj = s3_clnt.get_object(Bucket=bucket, Key="iris/batch_transform/iris_infer.csv")
payload_df = pd.read_csv(obj["Body"])

csv_buffer = io.StringIO()
payload_df.to_csv(csv_buffer, header=None, index=None)
payload_csv_text = payload_file.getvalue()

response = sgmkr_runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="text/csv", Body=payload_csv_text
)
print(response["Body"].read().decode())


# #### Delete the endpoint

# In[62]:


sgmkr_clnt = boto3.client("sagemaker")


# In[63]:


sgmkr_clnt.delete_endpoint(EndpointName=endpoint_name)


# In[ ]:





# #### Batch Transform

# In[64]:


batch_ip = "s3://{}/{}/{}".format(bucket, prefix, "batch_transform")
batch_op = "s3://{}/{}/{}".format(bucket, prefix, "batch_transform")


# In[65]:


get_ipython().system('aws s3 ls {batch_ip}/ --recursive')


# In[67]:


transformer = xgb_estimator.transformer(
    instance_count=1, instance_type="ml.m4.xlarge", output_path=batch_op
)


# In[69]:


transformer.transform(
    data=batch_ip, data_type="S3Prefix", content_type="text/csv", split_type="Line"
)
transformer.wait()


# In[ ]:





# In[70]:


get_ipython().system('aws s3 ls {bucket}/{prefix}/batch_transform/ --recursive')


# In[74]:


get_ipython().system('aws s3 cp s3://{bucket}/{prefix}/batch_transform/iris_infer.csv.out .')


# In[76]:


get_ipython().system('head -n 5 iris_infer.csv.out')


# In[ ]:





# In[ ]:





# In[ ]:




