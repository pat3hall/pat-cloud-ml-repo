#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('Employee_errors.csv')
df.head()


# In[3]:


df.dtypes


# In[4]:


df['first_name'] = df['first_name'].str.lower()
df['first_name'].head()


# In[5]:


df['last_name'] = df['last_name'].str.lower()
df['last_name'].head()


# In[6]:


df['first_name'] = df['first_name'].str.replace(' ','')
df['last_name'] = df['last_name'].str.replace(' ','')
df['first_name'].head()


# In[7]:


df['first_name'] = df['first_name'].str.strip()
df['last_name'] = df['last_name'].str.strip()
df['first_name'].head()


# In[8]:


df['department'].unique()


# In[9]:


def fix_errors(e):
    if e == 'marketing':
        return 'Marketing'
    elif e == 'Saless':
        return 'Sales'
    elif e == 'Marketting':
        return 'Marketing'
    else :
        return e


# In[10]:


df['department'] = df['department'].apply(fix_errors)
df['department'].head()


# In[11]:


df['department'].unique()


# In[12]:


gender = {'Male': 'M', 'Female': 'F', 'male': 'M', 'female': 'F', 'unknown': 'U', 'M': 'M', 'F': 'F', 'U': 'U'}
df['gender'] = df['gender'].map(gender)
df['gender'].head()


# In[15]:


df['salary'] = df['salary'].round()
df['salary'].head()


# In[16]:


df.head()


# In[ ]:





# In[ ]:




