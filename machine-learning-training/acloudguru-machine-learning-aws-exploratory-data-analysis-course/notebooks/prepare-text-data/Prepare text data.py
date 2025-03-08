#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords


# In[2]:


nltk.download('stopwords')
nltk.download('punkt')


# In[3]:


from nltk.tokenize import word_tokenize


# In[4]:


text = "I am excited to take machine learning certification"
tkns = word_tokenize(text)
tkns


# In[5]:


tkns_sw = [word for word in tkns if not word in stopwords.words('english')]
tkns_sw


# In[6]:


default = stopwords.words('english')
default.append('I')
tkns_sw_custom = [word for word in tkns if not word in default]
tkns_sw_custom


# In[ ]:




