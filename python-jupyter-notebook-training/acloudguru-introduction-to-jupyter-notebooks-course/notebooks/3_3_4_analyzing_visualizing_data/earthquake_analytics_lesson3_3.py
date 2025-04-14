#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[6]:


# load earthquake dataset
df1 = pd.read_csv("earthquakes-dataset_lesson3_3.csv")
# verify data was loaded
# report number of columns and rows
print(f"df1.shape: {df1.shape}")
# report 1st 3 lines plus column labels
print("\ndf1.head(3):")
df1.head(3)


# In[7]:


date_demo = "21st of July 2000"
print (f"date_demo: {date_demo}")
date_demo_datetime = pd.to_datetime(date_demo)
print (f"date_demo_datetime: {date_demo_datetime}")


# In[8]:


# create Datetime field which merges 'Date' and 'Time' columns 
# Note: add "errors = 'coerce'" argument so that invalid parsing results in empty values instead of failing
df1['Datetime'] =  pd.to_datetime(df1['Date'] + ' ' + df1['Time'], errors = 'coerce')
# verify new Datetime column
df1.head(3)


# In[11]:


# display Datetime in 1st row
print (f"df1['Datetime'][0]: {df1['Datetime'][0]}")
# display some 1st datetime info: %A (day of the week), %a (short form of day of the week), %B (month)
df1['Datetime'][0].strftime('%A or %a and is %B')


# In[12]:


# set Datetime column as the index field
df1 = df1.set_index(['Datetime'])
# verify index change
df1.head(3)


# In[13]:


# check the datatypes set when CSV file was read in (sometimes not set correctly)
df1.dtypes


# In[14]:


# convert Depth column from floating point to integer (using astype() method)
df1['Depth'] = df1['Depth'].astype(int)
# verify change
df1.head(3)


# In[15]:


# re-read CSV file, this time, merging 'Date' and 'Time' columns
df1 = pd.read_csv("earthquakes-dataset_lesson3_3.csv", index_col= 0, parse_dates= [['Date', 'Time']])
# verify updated dataframe
df1.head(3)


# In[17]:


# check index type (object or datetime?)
df1.index


# In[19]:


# df1.index return 'object' type, so convert index to 'datetime' type
df1.index = pd.to_datetime(df1.index, errors='coerce')
# verify df.index converted to 'datetime' type
df1.index


# In[20]:


# used 'info()' to determine columns with null values
df1.info()


# In[21]:


# drop all the columns (axis=1) with null values in some rows using 'dropna()'
df1 = df1.dropna(axis = 1)
# check 1st 3 rows
df1.head(3)


# In[22]:


# use 'info()' to show remain colunms and if all rows with null values have been removed
df1.info()


# In[24]:


# create a new dataframe, df2, with only the most important columns
df2 = df1[['Latitude', 'Longitude', 'Type', 'Depth', 'Magnitude']]
# check results (3 rows) 'sample()' which randomly picks row
df2.sample(3)


# In[25]:


# use 'unique()' to check the unique values in the 'Type' column
df2.Type.unique()


# In[27]:


# determine earthquake with largest 'Magnitude' in dataset
df2[df2.Type == "Earthquake"].Magnitude.max()


# In[28]:


# determine earthquake with largest 'Depth' (reported in KM) in dataset
df2[df2.Type == "Earthquake"].Depth.max()


# In[30]:


# determine earthquake with smallest 'Depth' in dataset
df2[df2.Type == "Earthquake"].Depth.min()


# In[31]:


# How many earthquakes in dataset (exmine Type "Earthquake" in results )
df2.Type.value_counts()


# In[34]:


# determine average Magnitude of earthquakes in 12 month periods using 'resample()' with mean()
df2['Magnitude'][df2.Type == 'Earthquake'].resample('12M').mean()


# In[38]:


# determine std deviation earthquakes Magnitudes in 12 month periods using 'resample()' with std()
df2['Magnitude'][df2.Type == 'Earthquake'].resample('12M').std()


# In[41]:


# find when the largest (Magnitude = 9.1) earthquake occurred
df2.loc[df2['Magnitude'] == 9.1] 


# In[42]:


# find when the smallest Depth (Depth = -1.1) earthquake occurred
df2.loc[df2['Depth'] == -1.1] 


# In[43]:


# find out how many nuclear explosions per year since 1965 (in dataset)
df2.index.year[df2.Type == 'Nuclear Explosion'].value_counts().sort_index()


# In[ ]:




