#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().system('python --version')


# In[15]:


#pip install -q -U nb_black


# In[3]:


#%load_ext nb_black


# In[4]:


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from IPython.display import Image

import boto3
import logging

boto3.set_stream_logger(name="botocore.credentials", level=logging.WARNING)


# #### Iris Dataset

# In[6]:


Image(url="images/iris_1.png", width=800)  # , height=300)


# In[7]:


Image(url="images/iris_2.svg", width=800)  # , height=300)


# In[8]:


Image(url="images/iris_3.png", height=400)


# In[9]:


iris = datasets.load_iris()
type(iris)


# In[ ]:





# In[10]:


dir(iris)


# In[ ]:





# In[10]:


print(iris.DESCR)


# In[10]:


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["class"] = pd.Series(iris.target)
df.head()


# In[11]:


iris.target_names


# In[12]:


print(df.shape)


# In[13]:


df["class"].value_counts()


# In[14]:


cols = list(df.columns)
print(cols)
cols = [cols[-1]] + cols[:-1]
print(cols)
df = df[cols]
df.head()


# #### Train - Test split

# In[15]:


train_df, test_df = train_test_split(
    df, test_size=0.33, random_state=42, stratify=df["class"]
)


# In[16]:


train_df["class"].value_counts()


# In[17]:


test_df["class"].value_counts()


# In[18]:


train_df.to_csv("data/iris_train.csv", index=False, header=None)


# In[19]:


test_df.to_csv("data/iris_test.csv", index=False, header=None)


# In[20]:


infer_df = test_df.drop(columns=["class"])
infer_df.head()


# In[21]:


infer_df.to_csv("data/iris_infer.csv", index=False, header=None)


# In[22]:


get_ipython().system('head data/iris_train.csv')


# In[23]:


get_ipython().system('head data/iris_infer.csv')


# In[ ]:





# In[24]:


get_ipython().system('aws s3 cp data/iris_train.csv s3://sgmkr-course/iris/data/')


# In[25]:


get_ipython().system('aws s3 cp data/iris_test.csv s3://sgmkr-course/iris/data/')


# In[26]:


get_ipython().system('aws s3 cp data/iris_infer.csv s3://sgmkr-course/iris/batch_transform/')


# In[ ]:





# In[27]:


import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")


# In[28]:


sb.pairplot(df, hue="class", palette="deep")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # 
