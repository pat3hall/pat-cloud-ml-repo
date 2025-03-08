#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[11]:


employee_df = pd.read_csv('Employee.csv')
employee_df.head()


# In[12]:


scaler.fit(employee_df['salary'].values.reshape(-1,1))


# In[13]:


employee_df['scaled_salary'] = scaler.transform(employee_df['salary'].values.reshape(-1,1))
employee_df[['scaled_salary']].describe()


# In[5]:


from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler(clip=True)


# In[6]:


mmscaler.fit(employee_df['salary'].values.reshape(-1,1),)
employee_df['salary_minmax_scaled'] = mmscaler.transform(employee_df['salary'].values.reshape(-1,1))
employee_df[['salary_minmax_scaled']].describe()


# In[7]:


employee_df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[16]:


employee_df = pd.read_csv('Employee_transformation.csv')
employee_df.head()


# In[28]:


np.min(employee_df['salary'])


# In[29]:


np.max(employee_df['salary'])


# In[20]:


employee_df['log_salary'] = np.log(employee_df['salary'])
employee_df.head()


# In[30]:


np.min(employee_df['log_salary'])


# In[31]:


np.max(employee_df['log_salary'])


# In[25]:


plt.hist(employee_df['salary'], bins=100)


# In[26]:


plt.hist(employee_df['log_salary'], bins=100)


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


employee_df = pd.read_csv('Employee_transformation.csv')
employee_df.head()


# In[3]:


np.min(employee_df['salary'])


# In[4]:


np.max(employee_df['salary'])


# In[5]:


employee_df['log_salary'] = np.log(employee_df['salary'])
employee_df.head()


# In[6]:


np.min(employee_df['log_salary'])


# In[7]:


np.max(employee_df['log_salary'])


# In[8]:


plt.hist(employee_df['salary'], bins=100)


# In[9]:


plt.hist(employee_df['log_salary'], bins=100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




