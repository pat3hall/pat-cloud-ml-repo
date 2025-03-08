#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sci


# In[10]:


employee_df = pd.read_csv('Employee.csv')
employee_df.head()


# In[11]:


employee_df = employee_df[['age','salary']]
employee_df.head()


# In[12]:


dendrogram = sci.dendrogram(sci.linkage(employee_df, method='single'))
plt.show()


# In[13]:


dendrogram = sci.dendrogram(sci.linkage(employee_df, method='complete'))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




