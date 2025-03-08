#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


employee_df = pd.read_csv('Employee_outliers.csv')
employee_df.head()


# In[3]:


employee_df.info()


# In[4]:


employee_df.boxplot()
plt.show()


# In[5]:


sns.boxplot(employee_df, x='salary')
plt.show()


# In[6]:


q1 = employee_df['salary'].quantile(0.25)
q2 = employee_df['salary'].median()
q3 = employee_df['salary'].quantile(0.75)
iqr = q3 - q1
iqr


# In[7]:


minimum = (q1 - (1.5*iqr))
maximum = (q3 + (1.5*iqr))
maximum


# In[8]:


cond1 = employee_df['salary'] < minimum
cond2 = employee_df['salary'] > maximum
outliers = employee_df[cond1 | cond2]
outliers


# In[9]:


employee_df.drop(index=outliers.index, inplace=True)


# In[10]:


sns.boxplot(employee_df,x='salary')
plt.show()


# In[11]:


employee_df = pd.read_csv('Employee_outliers.csv')
employee_df.head()


# In[12]:


data = employee_df['salary']
data.head()


# In[13]:


mean = np.nanmean(data.tolist())
std = np.nanstd(data.tolist())
std


# In[14]:


zscore = (data - mean)/std
zscore.head()


# In[15]:


threshold = 3
data[(abs(zscore) > threshold)]


# In[ ]:




