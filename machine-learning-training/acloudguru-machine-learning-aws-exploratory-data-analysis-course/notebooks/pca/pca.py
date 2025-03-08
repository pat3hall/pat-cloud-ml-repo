#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()
X = iris.data


# In[3]:


X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# In[4]:


covariance_matrix = np.cov(X_std.T)


# In[5]:


covariance_matrix


# In[6]:


eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)


# In[7]:


sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]


# In[8]:


k = 2  
principal_components = sorted_eigenvectors[:, :k]


# In[10]:


transformed_data = np.dot(X_std, principal_components)


# In[11]:


import matplotlib.pyplot as plt
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=iris.target)


# In[ ]:





# In[ ]:




