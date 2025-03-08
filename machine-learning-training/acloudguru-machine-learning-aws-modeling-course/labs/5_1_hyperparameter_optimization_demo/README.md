5.1 Understanding Hyperparameter Tuning

Saved files:
    completed python jupyter notebook:
      demos.ipynb
    extracted python code from jupyter notebook:
      demos.py
    html view from completed jupyter notebook:
      demos.html

  Resource:
    Perform Hyperparameter Optimization - Demos
      downloaded to:
      labs/lesson5_1_hyperparameter_optimization_demo/demos.ipynb


  parameters vs hyperparameters
    parameters
      - parameters are learned during the training process.
      - optimized during the training process.
      - specific to a model
      - The accuracy of these parameters directly affect the model's predictions.
    Hyperparameters
      - hyperparameters are set before the training process,
      - they define the training process and the model structure,
        - such as its complexity and the speed at which it learns.
      - Hyperparameters require tuning to find the best combination that yields optimal model performance.
        - start with an initial set and then based on the model's performance


  Parameters and Hyperparameters in Machine Learning and Deep Learning
    https://towardsdatascience.com/parameters-and-hyperparameters-aa609601a9ac
    Hyperparameters
     - Hyperparameters are parameters whose values control the learning process and determine the values of model
       parameters that a learning algorithm ends up learning.
     - you choose and set hyperparameter values that your learning algorithm will use before the training of the model even begins.
     - In this light, hyperparameters are said to be external to the model because the model cannot change its values during learning/training.
     - Hyperparameters are used by the learning algorithm when it is learning but they are not part of the resulting model.
    Hyperparameters examples:
      - Train-test split ratio
      - Learning rate in optimization algorithms (e.g. gradient descent)
      - Choice of optimization algorithm (e.g., gradient descent, stochastic gradient descent, or Adam optimizer)
      - Choice of activation function in a neural network (nn) layer (e.g. Sigmoid, ReLU, Tanh)
      - The choice of cost or loss function the model will use
      - Number of hidden layers in a nn
    Parameters
      - Parameters on the other hand are internal to the model.
      - That is, they are learned or estimated purely from the data during training as the algorithm used tries to
        learn the mapping between the input features and the labels or targets.
      - Model training typically starts with parameters being initialized to some values (random values or set to zeros).
      - As training/learning progresses the initial values are updated using an optimization algorithm (e.g. gradient descent).
    Parameter Examples
      - The coefficients (or weights) of linear and logistic regression models.
      - Weights and biases of a nn
      - The cluster centroids in clustering

  hyperparameter tuning approaches
    Grid Search.
      - Grid search is an exhaustive search method that evaluates every possible combination of the specified
        hyperparameter values
      - Only categorical parameters are supported when using the grid search strategy.
      - The model is trained and evaluated for each combination of hyperparameters in the grid using strategies
        like cross-validation.
      - The model is then predicted using evaluation metric like accuracy and F1 score.
      - the combination that yields a best performance according to the evaluation metric is selected as the optimal
        set of hyperparameters.
    Random Search
      - samples hyperparameter values from specific distributions and evaluates a set number of random combinations.
      - more efficient than Grid Search
      - For each such combination, the model is trained and evaluated using strategies like cross-validation
      - the model with best performance metrics is selected
      - offers a better trade-off between exploration and computational cost as compared to grid search
    Bayesian Optimization
      - an advanced strategy for hyperparameter tuning that uses probabilistic model of the cost function
        or the objective function and using it to select the most promising hyperparameters to evaluate.
      - more efficient than grid search and random search, because it intelligently selects the next set of
        hyperparameters to evaluate based on the previous results.
      - often finds better hyperparameters with fewer evaluations
      - this technique is more complex to understand and implement compared to grid search and random search.

    Amazon SageMaker Automatic Model Tuning (AMT)
       - SageMaker alternatively refers to hyperparameter tuning as Automatic Model Tuning (AMT)
       - AMT uses the algorithm and ranges of hyperparameters that you specify.
       - It then chooses the hyperparameter values that creates a model that performs the best as measured by
         a metric that you choose.


    Code: Perform Hyperparameter Optimization - Demos
          For Iris dataset, Uses sklearn GridSearchCV and RandomSearchCV to find the best DecisionTree hyperparameters

      >>> import numpy as np
      >>> from sklearn.datasets import load_iris
      >>> from sklearn.tree import DecisionTreeClassifier
      >>> from sklearn.model_selection import GridSearchCV, train_test_split
      >>> from sklearn.metrics import accuracy_score


      >>> data = load_iris()
      >>> X, y = data.data, data.target
      >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


      >>> #3 Define the model and setup the hyperparameter ranges

      >>> tree = DecisionTreeClassifier()

      >>> param_grid = {
      >>>     'max_depth': [None, 10, 20, 30, 40, 50],         # Maximum depth of the tree
      >>>     'min_samples_split': [2, 5, 10],                 # Minimum number of samples required to split an internal node
      >>>     'min_samples_leaf': [1, 2, 4],                   # Minimum number of samples required to be at a leaf node
      >>> }


      >>> # Step 4: Set up Grid Search with cross-validation
      >>> # n_jobs=-1 -> use all CPUs during the model process
      >>> grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)


      >>> # Step 5: Fit the Grid Search to the training data
      >>> grid_search.fit(X_train, y_train)


      >>> # Step 6: Evaluate the best model
      >>> print("Best hyperparameters found:")
      >>> print(grid_search.best_params_)

      >>> best_dt = grid_search.best_estimator_
      >>> y_predict = best_dt.predict(X_test)
      >>> print("\nValidation Accuracy:", accuracy_score(y_test, y_predict))


      >>> from sklearn.model_selection import RandomizedSearchCV

      >>> # Step 7: Set up Randomized Search with cross-validation
          # n_iter -> controls the number of different combinations of hyperparameters that will be tried during the tuning process.
          #        -> only 6 x 3 x 3 =54 possible combinations of hyperparameters, 54 iteration will be tried instead 100
      >>> random_search = RandomizedSearchCV(estimator=tree, param_distributions=param_grid, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)


      >>> # Step 8: Fit the Randomized Search to the training data
      >>> random_search.fit(X_train, y_train)


      >>> # Step 9: Evaluate the best model
      >>> print("Best hyperparameters found:")
      >>> print(random_search.best_params_)

      >>> best_dt = random_search.best_estimator_
      >>> y_pred = best_dt.predict(X_test)
      >>> print("\nValidation Accuracy:", accuracy_score(y_test, y_pred))



