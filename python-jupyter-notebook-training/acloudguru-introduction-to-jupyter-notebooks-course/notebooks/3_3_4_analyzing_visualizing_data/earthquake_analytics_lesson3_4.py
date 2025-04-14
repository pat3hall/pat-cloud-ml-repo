# In[1]:
import numpy as np
import pandas as pd

# In[2]:
# load earthquake dataset
df1 = pd.read_csv("earthquakes-dataset_lesson3_3.csv")
# verify data was loaded
# report number of columns and rows
print(f"df1.shape: {df1.shape}")
# report 1st 3 lines plus column labels
print("\ndf1.head(3):")
df1.head(3)

# In[3]:
date_demo = "21st of July 2000"
print (f"date_demo: {date_demo}")
date_demo_datetime = pd.to_datetime(date_demo)
print (f"date_demo_datetime: {date_demo_datetime}")

# In[4]:
# create Datetime field which merges 'Date' and 'Time' columns 
# Note: add "errors = 'coerce'" argument so that invalid parsing results in empty values instead of failing
df1['Datetime'] =  pd.to_datetime(df1['Date'] + ' ' + df1['Time'], errors = 'coerce')
# verify new Datetime column
df1.head(3)

# In[5]:
# display Datetime in 1st row
print (f"df1['Datetime'][0]: {df1['Datetime'][0]}")
# display some 1st datetime info: %A (day of the week), %a (short form of day of the week), %B (month)
df1['Datetime'][0].strftime('%A or %a and is %B')

# In[6]:
# set Datetime column as the index field
df1 = df1.set_index(['Datetime'])
# verify index change
df1.head(3)

# In[7]:
# check the datatypes set when CSV file was read in (sometimes not set correctly)
df1.dtypes

# In[8]:
# convert Depth column from floating point to integer (using astype() method)
df1['Depth'] = df1['Depth'].astype(int)
# verify change
df1.head(3)

# In[9]:
# re-read CSV file, this time, merging 'Date' and 'Time' columns
df1 = pd.read_csv("earthquakes-dataset_lesson3_3.csv", index_col= 0, parse_dates= [['Date', 'Time']])
# verify updated dataframe
df1.head(3)

# In[10]:
# check index type (object or datetime?)
df1.index

# In[11]:
# df1.index return 'object' type, so convert index to 'datetime' type
df1.index = pd.to_datetime(df1.index, errors='coerce')
# verify df.index converted to 'datetime' type
df1.index

# In[12]:
# used 'info()' to determine columns with null values
df1.info()

# In[13]:
# drop all the columns (axis=1) with null values in some rows using 'dropna()'
df1 = df1.dropna(axis = 1)
# check 1st 3 rows
df1.head(3)

# In[14]:
# use 'info()' to show remain colunms and if all rows with null values have been removed
df1.info()

# In[15]:
# create a new dataframe, df2, with only the most important columns
df2 = df1[['Latitude', 'Longitude', 'Type', 'Depth', 'Magnitude']]
# check results (3 rows) 'sample()' which randomly picks row
df2.sample(3)

# In[16]:
# use 'unique()' to check the unique values in the 'Type' column
df2.Type.unique()

# In[17]:
# determine earthquake with largest 'Magnitude' in dataset
df2[df2.Type == "Earthquake"].Magnitude.max()

# In[18]:
# determine earthquake with largest 'Depth' (reported in KM) in dataset
df2[df2.Type == "Earthquake"].Depth.max()

# In[19]:
# determine earthquake with smallest 'Depth' in dataset
df2[df2.Type == "Earthquake"].Depth.min()

# In[20]:
# How many earthquakes in dataset (exmine Type "Earthquake" in results )
df2.Type.value_counts()

# In[21]:
# determine average Magnitude of earthquakes in 12 month periods using 'resample()' with mean()
df2['Magnitude'][df2.Type == 'Earthquake'].resample('12M').mean()

# In[22]:
# determine std deviation earthquakes Magnitudes in 12 month periods using 'resample()' with std()
df2['Magnitude'][df2.Type == 'Earthquake'].resample('12M').std()

# In[23]:
# find when the largest (Magnitude = 9.1) earthquake occurred
df2.loc[df2['Magnitude'] == 9.1] 

# In[24]:
# find when the smallest Depth (Depth = -1.1) earthquake occurred
df2.loc[df2['Depth'] == -1.1] 

# In[25]:
# find out how many nuclear explosions per year since 1965 (in dataset)
df2.index.year[df2.Type == 'Nuclear Explosion'].value_counts().sort_index()

# In[26]:
get_ipython().system('pip3 install matplotlib')

# In[27]:
import matplotlib as mpl
import matplotlib.pyplot as plt

# In[28]:
# enable embedding static matplotlib plots in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# another similiar option is: %matplotlib notebook

# In[29]:
# create a histogram of Magnitude of earthquakes
plt.hist(df2['Magnitude'])
# add labels and title
plt.xlabel('Magnitude')
plt.ylabel('Number of Earthquakes')
plt.title('1969-2016 Earthquakes')

# In[30]:
# create a histogram of Magnitude of earthquakes - specify number bins and add black edgecolor
plt.hist(df2['Magnitude'], bins=10, edgecolor = "black")
# add labels and title
plt.xlabel('Magnitude')
plt.ylabel('Number of Earthquakes')
plt.title('1969-2016 Earthquakes')

# In[31]:
#  update histogram- explicitly specify 11 bins between Magnitudes of 6 and 7
plt.hist(df2['Magnitude'], bins=[6,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7], edgecolor = "black")
# add labels and title
plt.xlabel('Magnitude')
plt.ylabel('Number of Earthquakes')
plt.title('1969-2016 Earthquakes')

# In[32]:
# define Earthqaake Magnitude histogram ploting function with number of bins as input
def myplot(bins):
    # create a histogram of Magnitude of earthquakes - pass in the number bins and add black edgecolor
    plt.hist(df2['Magnitude'], bins=bins, edgecolor = "black")
    # add labels and title
    plt.xlabel('Magnitude')
    plt.ylabel('Number of Earthquakes')
    plt.title(f'1969-2016 Earthquakes - bins:{bins}')

# In[33]:
# plot histogram with 20 bins
myplot(20)

# In[34]:
# create a histogram of Magnitude of earthquakes - with 20 bins and between 8 and 9.1
plt.hist(df2['Magnitude'], bins=20, edgecolor = "black", range = [8, 9.1])
# add labels and title
plt.xlabel('Magnitude')
plt.ylabel('Number of Earthquakes')
plt.title('1969-2016 Earthquakes - 20 bins between 8 and 9.1')

# In[35]:
# create a earthquake Magnitude line chart showing largest earthquake during each year (using resample for 1 year + max)
plt.plot(df2['Magnitude'][df2.Type == "Earthquake"].resample('1Y').max())

# In[36]:
# check shape of bar expected inputs
df2.index.year.unique().shape

# In[37]:
df2.index.year[df2.Type == 'Earthquake'].value_counts().sort_index().shape

# In[38]:
# mismatch shape's reported (53,) vs (52,) - may have null values - need to find null values
df2.index.year.isnull().sum()

# In[39]:
# reported 3 null values - need find rows
df2.loc[df2.index.year.isnull()]

# In[40]:
# found 3 rows with null Date_Time index values - need reset index, drop Date_Time null values, then set index to 'Date_Time'
df2 = df2.reset_index().dropna().set_index('Date_Time')

# In[41]:
# create a earthquake Magnitude bar chart showing number of earthquakes per year
plt.bar(df2.index.year.unique(),df2.index.year[df2.Type == 'Earthquake'].value_counts().sort_index())
plt.title('Earthquakes per year')

# In[43]:
# previous bar chart showed earthquakes per year were increasing
# Now, check for nuclear explosions per year
plt.bar(df2[df2.Type == 'Nuclear Explosion'].index.year.unique(), 
        df2.index.year[df2.Type == 'Nuclear Explosion'].value_counts().sort_index())
plt.title('Nuclear Explosions per year')

# In[51]:
# Create a scatter plot for earthquakes per year
plt.scatter(df2[df2.Type == 'Earthquake'].index.year.unique(), 
        df2.index.year[df2.Type == 'Earthquake'].value_counts().sort_index())
plt.title('Earthquakes per year scatter plot')

# In[83]:
# Create a bubble chart (scatter plot with depth variable [bubble size]) for earthquakes per year
plt.scatter(df2[df2.Type == 'Earthquake'].index.year.unique(), 
        df2.index.year[df2.Type == 'Earthquake'].value_counts().sort_index(), 
        df2['Depth'][df2.Type == "Earthquake"].resample('1Y').max())
plt.title('Earthquakes max depth per year Bubble Chart')

# In[84]:
# Create a bubble chart (scatter plot with depth variable [bubble size]) for earthquakes per year
# change bubble color to 'green' and alpha blending factor
plt.scatter(df2[df2.Type == 'Earthquake'].index.year.unique(), 
        df2.index.year[df2.Type == 'Earthquake'].value_counts().sort_index(), 
        df2['Depth'][df2.Type == "Earthquake"].resample('1Y').mean(), 'green', alpha=0.7 )
plt.title('Earthquakes mean depth per year Bubble Chart in green with alpha=0.7')

# In[86]:
# scatter plot for Magnitude vs depth
plt.scatter(df2['Magnitude'], df2['Depth'])
plt.title('Magnitude vs Depth Scatter Plot')

# In[89]:
# map earthquakes by longitude and latitude
# set figure size to width: 19 inches x height: 10 inches (default 6.4 x 4,8)
plt.figure(figsize= (19,10))
# plt.scatter(<x>->Longitude, <y>->Latitude, <size>->Magnitude*10, <color>->Depth)
plt.scatter(df2['Longitude'], df2['Latitude'], df2['Magnitude'] * 10, df2['Depth'])
