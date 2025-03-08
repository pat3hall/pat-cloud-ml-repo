#!/usr/bin/env python
# coding: utf-8

# ![A Cloud Guru](acg_logo.png)
# <hr/>

# <center><h1>Comparing Machine Learning Algorithms on a Single Dataset using Amazon SageMaker</h1></center>

# # Introduction
# 
# In this lab, you will learn how to import a dataset, split it into training and test data, initialize multiple algorithms, train them, predict for test data and compare the metrics against the test data.

# # How to Use This Lab
# 
# Most of the code is provided for you in this lab as our solution to the tasks presented. Some of the cells are left empty with a #TODO header and its your turn to fill in the empty code. You can always use our lab guide if you are stuck.

# # 1) Import the Libraries

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


# # 2) Load the Dataset

# In[2]:


# Load the Dataset and create feature and target variables
california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.Series(california_housing.target, name='MedHouseVal')


# # 3) Split the Data

# In[3]:


# TODO: Use `train_test_split` function and split the data 80, 20 ratio. Assign the result to X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # 4) Initialize and Train the Algorithms

# In[4]:


# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[5]:


# TODO: Train a Random Forest Regression algorithm. pass two parameters n_estimators with a value of 100 and random_state with a value 42.
# TODO: Assign the result to rf_model. Fit the training data similar to linear regression algorithm.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[6]:


# TODO: Train a Ridge algorithm. This algorithm doesnt require any parameters.
# TODO: Assign the result to ridge. Fit the training data similar to linear regression algorithm.
ridge = Ridge()
ridge.fit(X_train, y_train)


# # 5) Make Predictions

# In[7]:


# Make predictions on the testing set
lr_predictions = lr_model.predict(X_test)


# In[8]:


# TODO: Make predictions on the random forest regression model and ridge model. Assign the predictions to `rf_predictions` and `ridge_predictions`
rf_predictions = rf_model.predict(X_test)
ridge_predictions = ridge.predict(X_test)


# # 6) Evaluate the Model

# In[9]:


# Evaluate the performance of all the models
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
lr_rme = root_mean_squared_error(y_test, lr_predictions)

#TODO: In the same fashion, fetch MAE (mean absolute error), R2 Score and RMSE (Root mean squared error) for the remaining two models.
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
rf_rme = root_mean_squared_error(y_test, rf_predictions)

ri_mae = mean_absolute_error(y_test, ridge_predictions)
ri_r2 = r2_score(y_test, ridge_predictions)
ri_rme = root_mean_squared_error(y_test, ridge_predictions)


# # 7) Validate the Output

# In[10]:


print("Linear Regression:")
print(f"Mean Absolute Error: {lr_mae}")
print(f"R-squared: {lr_r2}")
print(f"Root Mean Squared Error: {lr_rme}")

print("\nRandom Forest Regression:")
print(f"Mean Absolute Error: {rf_mae}")
print(f"R-squared: {rf_r2}")
print(f"Root Mean Squared Error: {rf_rme}")

print("\nRidge Regression:")
print(f"Mean Absolute Error: {ri_mae}")
print(f"R-squared: {ri_r2}")
print(f"Root Mean Squared Error: {ri_rme}")


# In[ ]:




