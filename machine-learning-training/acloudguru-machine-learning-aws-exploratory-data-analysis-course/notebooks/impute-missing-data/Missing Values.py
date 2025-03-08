#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Employee_missing.csv')
df.head()


# In[3]:


get_ipython().system('pip install missingno')


# In[4]:


import missingno as mn
mn.matrix(df)


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df['department'].mode()


# In[8]:


df['department'] = df['department'].fillna(df['department'].mode()[0])


# In[9]:


df.isnull().sum()


# In[11]:


df['age'] = df['age'].fillna(df['age'].mean())


# In[15]:


df['age'] = df['age'].astype(int)


# In[16]:


df.head()


# In[17]:


from sklearn.impute import KNNImputer
impute = KNNImputer(n_neighbors=2)


# In[27]:


df = pd.read_csv('Employee_missing.csv')
df.head()


# In[28]:


column = df['age']
df_imputed = impute.fit_transform(column.values.reshape(-1,1))
df_imputed = df_imputed.astype('int')


# In[29]:


df_imp = df.copy()


# In[30]:


df_imp['age'] = df_imputed.flatten()


# In[31]:


df_imp.head()


# In[ ]:




