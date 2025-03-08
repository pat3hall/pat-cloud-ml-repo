#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import boto3


# #### Methods

# In[2]:


def display_results(response:dict, user_profile:str=None) -> None:
    list_1 = []
    list_2 = []
    for i,item in enumerate(response['ResultItems']):
        title = item['DocumentTitle']['Text']
        score = item['ScoreAttributes']['ScoreConfidence']
        for attr in item['DocumentAttributes']:
            if (attr['Key'] == 'user_profile'):
                if  attr['Value']['StringValue'] == user_profile:
                    list_1.append(f'{i}. [{score}] [{attr["Value"]["StringValue"]}] {title}')
                else:
                    list_2.append(f'{i}. [{score}] [{attr["Value"]["StringValue"]}] {title}')
                break
            else:
                continue
    results = list_1 + list_2
    _ = [print(item) for item in results]


# ## Query Parameters

# In[5]:


kendra = boto3.client("kendra")

index_id = "c6b5e9c5-9d95-4385-bfa9-e1d2412fab16"
query = "boots please"
user_profile = "consumer"


# ## Query Index
# 
# SDK Guide: https://docs.aws.amazon.com/kendra/latest/dg/searching-example.html<b><b>
# 
# Request Parameters: https://docs.aws.amazon.com/kendra/latest/dg/API_Query.html
# 

# In[6]:


response = kendra.query(
    QueryText = query,
    IndexId = index_id
)


# #### Response Example

# In[7]:


response.keys()


# In[8]:


response['TotalNumberOfResults']


# In[9]:


response['ResultItems'][0]


# ### Results by Relevance

# In[10]:


display_results(response)


# ### Results by User Profile

# In[11]:


display_results(response, user_profile)


# In[ ]:




