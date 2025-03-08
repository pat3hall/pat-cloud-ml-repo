#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import os
import boto3
import json
import numpy as np
import pandas as pd


# # Data Processing

# ### Convert CSV into JSONs

# In[2]:


columns = ['name','description','itemid','colorname','Classification']

df = pd.read_csv('productsclassified.csv')[columns]
df.head()


# In[3]:


df['itemid'] = df['itemid'].apply(lambda x: x.replace('"','').replace("'",'').replace('[','').replace(']',''))
df['itemid'] = df['itemid'].apply(lambda x: x.split(',')[0] if len(x)>0 else 0)
df.head()


# In[4]:


df['colorname'] = df['colorname'].apply(lambda x: x.replace('"','').replace("'",'').replace('[','').replace(']',''))
df['colorname'] = df['colorname'].apply(lambda x: x.split(',') if len(x)>0 else [])
df.head()


# In[5]:


df['description'] = df['description'].apply(lambda x: x.replace('"','').replace("'",'').replace('[','').replace(']',''))
df['description'] = df['description'].apply(lambda x: x.split(',')[0] if len(x)>0 else '')
df.head()


# In[6]:


df = df.rename(columns={'name':'product','description':'description','itemid':'id','colorname':'colors','Classification':'category'})
df.head()


# ### Attach Mock User Profile

# In[7]:


user_profiles = ['reseller', 'consumer', 'distributer']
df['user_profile'] = np.random.choice(user_profiles, len(df))
df.head()


# ### Store Documents in S3
# 
# 1. Save and upload the Document-JSON to S3
# 2. Create, save, and upload the metadata-JSON to S3 in a folder

# In[9]:


bucket = 'pat-demo-bkt'
s3 = boto3.resource('s3')


for i,row in df.iterrows():
    # Process Document
    filename = f'doc_{i}_prod_{row.id}.json'
    js = {'id':row.id, 'product':row['product'], 'description':row.description}
    #js = {'id':row.id, 'product':row['product'], 'description':row.description, 'colors':row.colors, 'category':row.category}
    json.dump(js, open(f'jsons/{filename}', 'w'), indent=4)
    s3.meta.client.upload_file(f'jsons/{filename}', bucket, filename)
    
    # Process Metadata
    colors_list = row.colors[:10]
    filename = f'{filename}.metadata.json'
    md = {
        "DocumentId": f'doc_{i}_prod_{row.id}.json',
        "Attributes": {
            "_category": row.category,
            'colors':colors_list  if len(colors_list) > 0 else ['No color was provided'],
            #"description": row.description if len(row.description) > 10 else 'There was no description provided.',
            "user_profile": row.user_profile
        },
        "Title": row['product'],
    }
    json.dump(md, open(f'jsons/{filename}', 'w'), indent=4)
    s3.meta.client.upload_file(f'jsons/{filename}', bucket, f'metadata/{filename}')


# In[ ]:




