#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from skimage.io import imread, imshow


# In[3]:


#image = imread('blocks-letters_Blue.png', as_gray=True) 
# "block-letters_blue.png" was not provided so using:
image = imread('abc_blocks.png', as_gray=True)


# In[5]:


height, width = image.shape


# In[6]:


imshow(image)


# In[7]:


features = height * width
print(f'features: {features}, height: {height}, width: {width}')


# In[8]:


features


# In[9]:


image = imread('abc_blocks.png') 


# In[10]:


image.shape


# In[11]:


from skimage.filters import prewitt_h,prewitt_v


# In[12]:


image = imread('abc_blocks.png', as_gray=True) 


# In[13]:


edges_prewitt_horizontal = prewitt_h(image)


# In[14]:


edges_prewitt_vertical = prewitt_v(image)


# In[15]:


imshow(edges_prewitt_vertical, cmap='gray')


# In[16]:


imshow(edges_prewitt_horizontal, cmap='gray')


# In[ ]:




