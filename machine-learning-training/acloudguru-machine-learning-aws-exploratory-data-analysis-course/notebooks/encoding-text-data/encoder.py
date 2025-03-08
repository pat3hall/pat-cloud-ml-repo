#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np


# In[31]:


employee_df = pd.read_csv('Employee_encoding.csv')
employee_df.head()


# In[32]:


from sklearn.preprocessing import LabelEncoder
title_encoder = LabelEncoder()
employee_df['title'].unique()


# In[33]:


title_encoder.fit(employee_df['title'])


# In[34]:


title_encoder.classes_


# In[35]:


title_encoder.classes_ = np.array(['developer','senior developer','manager','vp'])


# In[36]:


employee_df['encoded_title'] = title_encoder.transform(employee_df['title'])


# In[37]:


employee_df[['title', 'encoded_title']]


# In[38]:


from sklearn.preprocessing import OneHotEncoder
gender_encoder = OneHotEncoder()


# In[39]:


employee_df['gender'].unique()


# In[40]:


gender_encoder.fit(employee_df['gender'].values.reshape(-1,1))


# In[41]:


gender_encoder.categories_


# In[42]:


transform = gender_encoder.transform(employee_df['gender'].values.reshape(-1,1))
employee_df1 = pd.DataFrame(transform.todense(), columns=gender_encoder.categories_)
employee_df1


# In[43]:


combined_df = employee_df.join(employee_df1)
combined_df


# In[ ]:




