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


# In[8]:


print(f'Q1: {q1}, Median: {q2}, Q3: {q3}, IQR: {iqr}')


# In[9]:


minimum = (q1 - (1.5*iqr))
maximum = (q3 + (1.5*iqr))
print (f'minium: {minimum},  maximum: {maximum}')


# In[10]:


cond1 = employee_df['salary'] < minimum
cond2 = employee_df['salary'] > maximum
outliers = employee_df[cond1 | cond2]
outliers


# In[11]:


employee_df.drop(index=outliers.index, inplace=True)


# In[12]:


sns.boxplot(employee_df,x='salary')
plt.show()


# In[13]:


employee_df = pd.read_csv('Employee_outliers.csv')
employee_df.head()


# In[15]:


data = employee_df['salary']
data.head()


# In[16]:


mean = np.nanmean(data.tolist())
std = np.nanstd(data.tolist())
print (f'std: {std}, mean: {mean}')


# In[18]:


zscore = (data - mean)/std
zscore.head()


# In[19]:


threshold = 3
data[(abs(zscore) > threshold)]


# In[ ]:




