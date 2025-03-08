#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt


# In[8]:


data = np.array([10, 11, 12, 13, 14, 15, 16, 20, 25, 30, 35, 40, 45, 50, 55]).reshape(-1, 1)


# In[9]:


kbins = KBinsDiscretizer(n_bins=5, strategy='quantile', encode='ordinal')


# In[10]:


bins_EF = kbins.fit_transform(data)


# In[11]:


bins_EF


# In[12]:


plt.hist(bins_EF)
plt.show()


# In[20]:


kbins_width = KBinsDiscretizer(n_bins=5, strategy='uniform', encode='ordinal', subsample=None)


# In[21]:


bins_EW=kbins_width.fit_transform(data)


# In[22]:


bins_EW


# In[17]:


plt.hist(bins_EW)
plt.show()


# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt


# In[2]:


data = np.array([10, 11, 12, 13, 14, 15, 16, 20, 25, 30, 35, 40, 45, 50, 55]).reshape(-1, 1)


# In[3]:


kbins = KBinsDiscretizer(n_bins=5, strategy='quantile', encode='ordinal')


# In[4]:


bins_EF = kbins.fit_transform(data)


# In[5]:


bins_EF


# In[6]:


plt.hist(bins_EF)
plt.show()


# In[7]:


kbins_width = KBinsDiscretizer(n_bins=5, strategy='uniform', encode='ordinal', subsample=None)


# In[8]:


bins_EW=kbins_width.fit_transform(data)


# In[9]:


bins_EW


# In[10]:


plt.hist(bins_EW)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




