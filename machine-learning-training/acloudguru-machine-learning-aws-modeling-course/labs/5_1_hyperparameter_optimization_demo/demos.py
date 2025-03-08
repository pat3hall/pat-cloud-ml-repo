#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


#3 Define the model and setup the hyperparameter ranges

tree = DecisionTreeClassifier()

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],         # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],                 # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],                   # Minimum number of samples required to be at a leaf node
}


# In[4]:


# Step 4: Set up Grid Search with cross-validation
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)


# In[5]:


# Step 5: Fit the Grid Search to the training data
grid_search.fit(X_train, y_train)


# In[6]:


# Step 6: Evaluate the best model
print("Best hyperparameters found:")
print(grid_search.best_params_)

best_dt = grid_search.best_estimator_
y_predict = best_dt.predict(X_test)
print("\nValidation Accuracy:", accuracy_score(y_test, y_predict))


# In[11]:


from sklearn.model_selection import RandomizedSearchCV

# Step 7: Set up Randomized Search with cross-validation
random_search = RandomizedSearchCV(estimator=tree, param_distributions=param_grid, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)


# In[12]:


# Step 8: Fit the Randomized Search to the training data
random_search.fit(X_train, y_train)


# In[13]:


# Step 9: Evaluate the best model
print("Best hyperparameters found:")
print(random_search.best_params_)

best_dt = random_search.best_estimator_
y_pred = best_dt.predict(X_test)
print("\nValidation Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:




