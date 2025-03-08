#!/usr/bin/env python
# coding: utf-8

# ![A Cloud Guru](acg_logo.png)
# <hr/>

# <center><h1>Visualizing Data with Python Libraries and Amazon SageMaker</h1></center>

# # Introduction
# In this lab, you will learn how to visualize data using various charts like histogram, bar chart, scatter plot and box plot. The provided dataset contains a list of employees with their gender, age, salary, and the department that contains both numerical and categorical data.

# # How to Use This Lab
# We have already imported the required libraries and imported the dataset for you. You need to use the seaborn library to visualize the data. You need to choose the right type of chart to address the given business problem. Please pay attention to the cells with a #TODO header where you need to enter the code. You can always use our lab guide if you are stuck.

# # 1) Import the Libraries

# In[1]:


# import required libraries.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 2) Read the Data

# In[2]:


# Read the dataset and display first few rows.

employee_df = pd.read_csv('Employee.csv')
employee_df.head()


# # 3) Visualizing Univariate Data

# In[4]:


# Show the distribution of all the employee's age in your oganization

sns.histplot(data=employee_df, x='age')
plt.show()


# In[5]:


# Show the number of employees in each department in your organization.
# TODO: Use the matplotlib's "xlabel", "ylabel", and "title" function 
# to set the x-axis as "Department", y-axis as "Employee Count" and title as "Distribution of Employees by Department"
sns.countplot(data=employee_df, x='department')
plt.xlabel('Department')
plt.ylabel('Employee Count')
plt.title('Distribution of employees by department')
plt.show()


# In[6]:


# Show the number of male and female employees in ech department in your organization
# TODO: Modify the countplot function to show the gender. 
# Tip: Use the hue attribute that will create separate bars for each category on the x-axis.

sns.countplot(data=employee_df, x='department', hue='gender')
plt.show()


# # 4) Visualizing Bivariate Data

# In[7]:


# bivariate analysis of a numerical and categorical features.
# Show the salaries of all the employees by their departments and check if there are any salary discrepancies.

sns.barplot(data=employee_df, x='department', y='salary')
plt.show()


# In[8]:


# Comparison of continuous and categorical features
# Show the descriptive statistics of employee's salary across all the departments
sns.boxplot(data=employee_df, x='department', y='salary')
plt.show()


# In[9]:


# Relationship between two continuous features.
# Show the relationship between an employee's age and their salary.

sns.scatterplot(data=employee_df, x='age', y='salary')
plt.show()


# In[ ]:




