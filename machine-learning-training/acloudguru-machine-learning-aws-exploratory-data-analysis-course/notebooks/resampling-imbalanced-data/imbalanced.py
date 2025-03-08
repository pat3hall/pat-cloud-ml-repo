#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb


# In[2]:


df = pd.read_csv('Employee_imbalanced.csv')
df.head()


# In[3]:


sb.countplot(df['expired_pto'])


# In[7]:


pd.Series(df['expired_pto']).value_counts()


# In[8]:


from sklearn.utils import resample
unexpired_pto = df[(df['expired_pto']=='n')]
expired_pto = df[(df['expired_pto']=='y')]
oversample = resample(expired_pto, replace=True, n_samples=99, random_state=40)
df_oversample = pd.concat([oversample, unexpired_pto])


# In[9]:


df_oversample.info()


# In[6]:


sb.countplot(df_oversample['expired_pto'])


# In[10]:


pd.Series(df_oversample['expired_pto']).value_counts()


# In[11]:


pip install imblearn


# In[12]:


from imblearn.over_sampling import SMOTE
df2 = pd.read_csv("Employee_imbalanced.csv")
smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=3)
X = df2[['age', 'salary']]
y = df2['expired_pto']
X


# In[14]:


X_sm, y_sm = smote.fit_resample(X, y)
y_sm.count()


# In[15]:


oversampled = pd.concat([pd.DataFrame(y_sm), pd.DataFrame(X_sm)], axis=1)
oversampled.head()


# In[16]:


pd.Series(oversampled['expired_pto']).value_counts()


# In[17]:


pd.Series(df2['expired_pto']).value_counts()


# In[ ]:




