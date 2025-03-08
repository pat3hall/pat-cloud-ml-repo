#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%load_ext nb_black


# In[2]:


import json
import pandas as pd
import logging
from ipywidgets import Dropdown


# In[3]:


import boto3
import sagemaker


# In[4]:


boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# In[5]:


sess = sagemaker.Session()
region = sess.boto_region_name
print(region)


# In[6]:


# download JumpStart model_manifest file.
boto3.client("s3").download_file(
    f"jumpstart-cache-prod-{region}", "models_manifest.json", "models_manifest.json"
)
with open("models_manifest.json", "rb") as json_file:
    model_list = json.load(json_file)

print("number of models: ", len(model_list))


# In[7]:


model_df = pd.DataFrame(model_list)
model_df.sample(20)


# In[8]:


# filter-out all the Object Detection models from the manifest list.
od_models = []
for model in model_list:
    model_id = model["model_id"]
    if ("-od-" in model_id or "-od1-" in model_id) and model_id not in od_models:
        od_models.append(model_id)

print(f"Number of od models available for inference: {len(od_models)}")


# In[9]:


# display the model-ids in a dropdown to select a model for inference.
infer_model_dropdown = Dropdown(
    options=od_models,
    value="pytorch-od-nvidia-ssd",
    description="Select a model:",
    style={"description_width": "initial"},
    layout={"width": "max-content"},
)


# In[10]:


display(infer_model_dropdown)


# In[11]:


print(infer_model_dropdown.value)


# In[ ]:





# In[12]:


# filter-out all the Image Classification models from the manifest list.
ic_models = []
for model in model_list:
    model_id = model["model_id"]
    if ("-ic-" in model_id) and model_id not in ic_models:
        ic_models.append(model_id)

print(f"Number of ic models available for inference: {len(ic_models)}")


# In[13]:


# display the model-ids in a dropdown to select a model for inference.
infer_model_dropdown = Dropdown(
    options=ic_models,
    value="pytorch-ic-alexnet",
    description="Select a model:",
    style={"description_width": "initial"},
    layout={"width": "max-content"},
)


# In[14]:


display(infer_model_dropdown)


# In[15]:


print(infer_model_dropdown.value)


# In[ ]:





# In[ ]:




