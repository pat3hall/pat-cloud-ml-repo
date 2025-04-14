#!/usr/bin/env python
# coding: utf-8

# In[1]:
# import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# enable embedding static matplotlib plots in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# In[2]:
# suppress warning messages
import warnings
warnings.filterwarnings('ignore')

# In[3]:
# import nasa gistemp dataset (earth mean temperature by month from 1880 to 2019)
df1 = pd.read_csv("nasa_gistemp.csv")
# verify it was imported
df1.head(3)

# In[4]:
# set index to colunm 0 (Year) and skip row 1 (header)
df1 = pd.read_csv("nasa_gistemp.csv", index_col = 0, skiprows = 1)
# verify it was imported
df1.head(3)

# In[5]:
# check for null values using info()
df1.info()

# In[6]:
# shows 7 columns have non-floating point values
# verify no null values using isnull()
df1.isnull().sum()

# In[7]:
# no null values reported
# need to clean-up non-floating point values with a cleanup function
def cleanup(x):
    try:
        return float(x)
    except:
        return np.NaN

# In[8]:
# test out cleanup function on Aug column using 'apply' function
df1['Aug'].apply(cleanup)

# In[9]:
# create for loop to iterate through the colunn using forware and backward fitting
for columns in df1.columns:
    # replace non-floating point values with NaN
    df1[columns] = df1[columns].apply(cleanup)
    # replace NaN values with forward fill value (next row value)
    df1[columns].fillna(method = 'ffill', inplace=True)
    # (if no forward file value found,) replace NaN values with back fill value (last row value)
    df1[columns].fillna(method = 'bfill', inplace=True)

# check if it was cleaned up
df1

# In[10]:
# set figsize = (x,y), and xy and y labels, and plot GIS TEMP
plt.figure(figsize= (8,6))
plt.xlabel('Time')
plt.ylabel('Temperature Anomaly in Celsius')
plt.plot(df1)

# In[11]:
# use seaborn (sns) lineplot to plot Jun & Dec starting in '1913' (ford model T 1st year)
plt.figure(figsize= (10,8))
# loc[<index>, <column>]
sns.lineplot(data = [df1.loc['1913':,'Jun'],df1.loc['1913':,'Dec']])

# In[12]:
# use seaborn (sns) lineplot to plot Jun & Dec temp data with 0.6 linewidth
#  and disable dash lines (by default, up to 6 lines may use dashes)
plt.figure(figsize= (10,8))
# loc[<index>, <column>]
sns.lineplot(data = df1.loc[:,'Jan' : 'Dec'], linewidth = 0.6, dashes = False)

# In[13]:
# display available styles (since 'seaborn' does not work)
print(plt.style.available)

# In[14]:
# create subplots with style context set to seaborn-v0_8
# generate 12 subplots (1 per month) with 4 rows and 3 columns
# plot color=(<red>,<green>,<blue>,<alpha>) where the values are floating points between 0 and 1
with plt.style.context('seaborn-v0_8'):
    fig, axes = plt.subplots(4,3, figsize=(15,15))
    # super title, suptitle(), is title at top
    fig.suptitle("Temperature anomalies throughout the years")
    col = 0
    for i in range (4):
        for j in range(3):
            # create 3 subplots per row
            axes[i,j].plot(df1.index, df1.loc[:, df1.columns[col]], color = (0, col/12, col/12, 1))
            axes[i,j].set_title(df1.columns[col])
            col += 1
            

# In[15]:
# create a heat map - centered on "0" value (center=0)
# use integer column indexing (e.g. iloc())
plt.figure(figsize= (10,8))
sns.heatmap(df1.iloc[:,0:12], center = 0)

# In[16]:
# display seaborn colormap value for "RdBu_r" (Rd: red, Bu: Blue, _r: end with red))
sns.color_palette("RdBu_r")

# In[17]:
# create a heat map - centered on "0" value (center=0) plus specify "RdBu_r" color map (cmap)
# use integer column indexing (e.g. iloc())
plt.figure(figsize= (10,8))
sns.heatmap(df1.iloc[:,0:12], cmap = 'RdBu_r', center = 0)

# In[18]:
# heatmap using loc() to specify years: 2000 - 2019 from Jan - Dec 
#  plus annotate values to heatmap (annot=True)
plt.figure(figsize= (10,8))
sns.heatmap(df1.loc['2000':'2019','Jan' : 'Dec'], cmap = 'RdBu_r', center = 0, annot=True)

# In[19]:
# install plotlib via installing "chart_studio" - includes interactive graphics
#!pip3 install chart_studio

# In[20]:
# import plotly (chart_studio)
import plotly.graph_objects as go

# In[23]:
# re-do seaborn heatmap with Plotly heatmap which creates an interactive heatmap
#   dimensions: z: data, x: months, y: years
go.Figure(data = go.Heatmap(z = df1, x = df1.columns[0:12], y = df1.index, colorscale = 'rdbu', reversescale=True))

# In[ ]:
# use mouse to zoom in or out of interactive heatmap or use zoom and other options in upper left corner
# hover over anywhere on map to see actual values

# In[24]:
# create plotly interactive Scatter plot with a trace for each month
fig = go.Figure()
col = 0
for col in range(12):
    fig.add_trace(go.Scatter(x = df1.index, y = df1.loc[:, df1.columns[col]], name = df1.columns[col]))
    col += 1
fig.show()

# In[ ]:
# can turn off plots by clicking month labels on right side
