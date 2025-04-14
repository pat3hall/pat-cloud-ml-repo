#!/usr/bin/env python
# coding: utf-8

# In[3]:
get_ipython().system('pip3 install seaborn')

# In[4]:
# import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# enable embedding static matplotlib plots in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# In[6]:
# suppress warning messages
import warnings
warnings.filterwarnings('ignore')

# In[7]:
# create an array between 0 and 10 with 1 interval
x = np.linspace(0,10,11)
x

# In[13]:
# plot solid and dash lines with a legend
# Note: plot(<x>, <y>, <fmt>) where fmt: '-' : solid line, '--' dash line
plt.plot(x, x, '-', label = 'my x')
plt.plot(x, x-1, '--', label = 'my smaller x')
plt.legend()

# In[16]:
# plot solid and dash lines with a legend
# add title to legend, place legend in lower center, and remove legend frame
plt.plot(x, x, '-', label = 'my x')
plt.plot(x, x-1, '--', label = 'my smaller x')
plt.legend(title = "I am legend", loc = 'lower center', frameon = False)

# In[24]:
# plot solid and dash lines with a legend - dash line: x by sin(x)
# add title to legend, place legend in upper left
plt.plot(x, x, '-', label = 'my x')
plt.plot(x, np.sin(x), '--', label = 'my smaller x')
plt.legend(title = "I am legend + sin(x)", loc = 'upper left')
# plt.text(<x>,<y>, <str>, ha=<horizontalAlignment>) where x,y: anchor point of text (default: 'center'), 
#    <str>: text to display,  ha: 'left, 'right', or 'center' (for anchor point)
plt.text(8, 1, 'my favorite number', ha = 'left')

# In[26]:
# add plt.annotate() to plt.plot()
plt.plot(x, x, '-', label = 'my x')
plt.plot(x, np.sin(x), '--', label = 'my smaller x')
plt.legend(title = "I am legend + sin(x)", loc = 'upper left')
# plt.annotate(<text>, xy = (x1,y1), xytext = (x2,y2), arrowprops = dict(<arrowStyle>))
#    where x1,y1 -> arrow anchor point, x2,y2 -> text anchor point
plt.annotate('my first annotation', xy = (8, 1), xytext = (8,5), arrowprops=dict(arrowstyle = '->')) 

# In[37]:
# create 2  2 rows x 1 cols subplots 
# plt.subplot(<nrows>, <ncols>, <index>)  where nrows: number of rows, ncols: number of columns, 
#    index: subplot position on grid - starts at 1 in the upper left corner
plt.subplot(2,1,1)
plt.subplot(2,1,2)

# In[38]:
# create 2  2 rows x 2 cols subplots 
plt.subplot(2,2,1)
plt.subplot(2,2,2)

# In[39]:
# create 2  2 rows x 1 cols subplots with space between 
#    hspace: hight reserved space, wspace: width reserved space
plt.figure().subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.subplot(2,1,1)
plt.subplot(2,1,2)

# In[46]:
# The 'fig_map.png' image graphically shows the 'fig' and 'axes' returned values from plt.subplots()
# axes: the x,y labels for the subplots
from IPython.display import Image
Image(url = 'https://matplotlib.org/1.5.1/_images/fig_map.png')

# In[45]:
# create a grid of subplots using plt.subplots() method and share row and column labels
fig, axes = plt.subplots(2,2, sharex = 'col', sharey = 'row')
# add x by x plots to the 0x0 axes and 1x1 axes subplots
axes[0,0].plot(x,x,"-")
axes[1,1].plot(x,x,"--")

# In[51]:
# create a grid of subplots using plt.subplots() method and share row and column labels
fig, axes = plt.subplots(2,2, sharex = 'col', sharey = 'row')
# iterate through the 4 subplots and plot x+(10*i) by x+(5*j)) 
for i in range(2):
    for j in range(2):
        axes[i,j].plot(x+10*i,x+5*j, "-")

# In[55]:
# generate 100 random sample (or samples) from the “standard normal” distribution using randn().
y = np.random.randn(100)
# plot histogram of random sample
plt.hist(y)

# In[56]:
# replot histogram using 'grayscale' style.context and 'white' edgecolor
with plt.style.context('grayscale'):
    plt.hist(y, edgecolor ='white')

# In[58]:
# replot histogram using 'dark_background' style.context and 'black' edgecolor
with plt.style.context('dark_background'):
    plt.hist(y, edgecolor ='black')

# In[63]:
# replot histogram using 'seaborn' style.context and 'black' edgecolor
# Note: This fails: OSError: 'seaborn' is not a valid package style ...
with plt.style.context('seaborn'):
    plt.hist(y, edgecolor ='black')
