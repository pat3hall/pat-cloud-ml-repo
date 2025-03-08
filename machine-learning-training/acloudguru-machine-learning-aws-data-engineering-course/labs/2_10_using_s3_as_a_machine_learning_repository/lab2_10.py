#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('s3://my-ml-repo-bkt/Parking_Tags_Data_2022.000.csv')
df.head(11)


# In[4]:


df.drop(columns='location3')


# In[5]:


df.to_csv('s3://my-ml-repo-bkt/Result.csv')


# In[ ]:




