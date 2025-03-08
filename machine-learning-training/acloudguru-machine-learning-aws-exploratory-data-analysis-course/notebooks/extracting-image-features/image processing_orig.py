#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from skimage.io import imread, imshow


# In[2]:


image = imread('blocks-letters_Blue.png', as_gray=True) 


# In[5]:


height, width = image.shape


# In[4]:


imshow(image)


# In[6]:


features = height * width


# In[7]:


features


# In[10]:


image = imread('blocks-letters_Blue.png') 


# In[11]:


image.shape


# In[9]:


from skimage.filters import prewitt_h,prewitt_v


# In[16]:


image = imread('blocks-letters_Blue.png', as_gray=True) 


# In[18]:


edges_prewitt_horizontal = prewitt_h(image)


# In[19]:


edges_prewitt_vertical = prewitt_v(image)


# In[20]:


imshow(edges_prewitt_vertical, cmap='gray')


# In[ ]:




