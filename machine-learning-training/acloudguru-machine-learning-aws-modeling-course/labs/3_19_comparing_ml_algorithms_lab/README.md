3.19 Comparing Machine Learning Algorithms on a Single Dataset using Amazon SageMaker

Saved files:
    completed python jupyter notebook:
      compare_algorithms_finished.ipynb
    extracted python code from jupyter notebook:
      compare_algorithms_finished.py

About this lab

  Imagine you are the data engineer, and you have been assigned the task of finding an optimal ML algorithm
  by comparing multiple algorithms. This lab will take the California housing dataset and predict the median housing value.

  In this hands-on lab, you will learn how to train multiple regression algorithms, predict for test data and compare
  core regression metrics.

Learning objectives
  - Launch SageMaker Notebook
  - Load Libraries and Fetch the Data (California Housing data)
  - Train the Model with Multiple Algorithms (Scikit Linear Learn, Random Forest, Ridge models)
  - Predict and Compare

Solution
Launch SageMaker Notebook

    To avoid issues with the lab, open a new Incognito or Private browser window to log in to the lab. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region. If you are prompted to select a kernel, please choose conda_tensorflow2_p310.
    In the search bar, type "SageMaker" to search for the SageMaker service. Click on the Amazon SageMaker result to go directly to the SageMaker service.
    Expand Applications and IDEs and select Notebooks to display the notebook provided by the lab.
    Check to see if the notebook is marked as InService. If so, click on the Open Jupyter link under Actions.
    Click on the compare_algorithms.ipnyb file.

Load the Dataset and Split the Data

Note: If this is your first time running a notebook, each cell contains Python commands you can run independently.

    Click the first cell that imports the required Python libraries, and use the Run button at the top to execute the code.

    Note: A * inside the square braces indicates the code is running, and you will see a number once the execution is complete.

    This cell indicates all the libraries we will import from Pandas and sklearn. It may take a few minutes to complete the operation.

    The next cell fetches the housing dataset from sklearn library. After loading the dataset, we create feature variables (X) and target variables (Y). Click Run to execute this cell.

    Now that the data is imported, it must be split for training and testing purposes. Use the following code snippet and Run the cell to perform this operation.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Initialize and Train the algorithms

    The next cell initializes LinearRegression and applies fit operation on the training data we fetched in the previous step. Run this cell to train this algorithm.

    In the same fashion, lets initialize RandomForestregressor algorithm and apply fit function to train the model. Copy the following code and click Run

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    We will compare three algorithms in this lab. As the third algorithm, we will use Ridge. Use the following code to train this model.

    ridge = Ridge()
    ridge.fit(X_train, y_train)

Make Predictions

    In this section, we will use all the three trained models and predict values for test data. The first cell under this category uses LinearRegression to predict. Run this cell to complete the prediction.

    Use the following code snippet to run predictions on RandomForestRegressor and Ridge models.

    rf_predictions = rf_model.predict(X_test)
    ridge_predictions = ridge.predict(X_test)

Evaluate the Model

    The first cell under this section, evaluates the performance of these models using three metrics MAE (mean absolute error), R2 Score and RMSE (Root mean squared error).

    The cell contains code to fetch the metrics for LinearRegression predictions, but we want to fetch the metrics for all three models. Copy the following code snippet and add it to the bottom of the cell and Run it to fetch the metrics for all three models.

    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    rf_rme = root_mean_squared_error(y_test, rf_predictions)

    ri_mae = mean_absolute_error(y_test, ridge_predictions)
    ri_r2 = r2_score(y_test, ridge_predictions)
    ri_rme = root_mean_squared_error(y_test, ridge_predictions)

Validate the Output

    Highlight the cell under the section and click Run to print the metrics from all the three models.
    Based on the results, the Random Forest Regression model would be the best model to use in this case.

   --------------------------------------------
   code: Compare algorithm lab

      >>> # Comparing Machine Learning Algorithms on a Single Dataset using Amazon SageMaker

      >>> # # Introduction
      >>> #
      >>> # In this lab, you will learn how to import a dataset, split it into training and test data, initialize multiple algorithms, train them, predict for test data and compare the metrics against the test data.

      >>> # # How to Use This Lab
      >>> #
      >>> # Most of the code is provided for you in this lab as our solution to the tasks presented. Some of the cells are left empty with a #TODO header and its your turn to fill in the empty code. You can always use our lab guide if you are stuck.

      >>> # # 1) Import the Libraries

      >>> import pandas as pd
      >>> from sklearn.model_selection import train_test_split
      >>> from sklearn.datasets import fetch_california_housing
      >>> from sklearn.model_selection import train_test_split
      >>> from sklearn.linear_model import LinearRegression
      >>> from sklearn.linear_model import Ridge
      >>> from sklearn.ensemble import RandomForestRegressor
      >>> from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


      >>> # # 2) Load the Dataset

      >>> # Load the Dataset and create feature and target variables
      >>> california_housing = fetch_california_housing()
      >>> X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
      >>> y = pd.Series(california_housing.target, name='MedHouseVal')


      >>> # # 3) Split the Data

      >>> # TODO: Use `train_test_split` function and split the data 80, 20 ratio. Assign the result to X_train, X_test, y_train, y_test
      >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


      >>> # # 4) Initialize and Train the Algorithms

      >>> # Train a Linear Regression model
      >>> lr_model = LinearRegression()
      >>> lr_model.fit(X_train, y_train)


      >>> # TODO: Train a Random Forest Regression algorithm. pass two parameters n_estimators with a value of 100 and random_state with a value 42.
      >>> # TODO: Assign the result to rf_model. Fit the training data similar to linear regression algorithm.
      >>> rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
      >>> rf_model.fit(X_train, y_train)


      >>> # TODO: Train a Ridge algorithm. This algorithm doesnt require any parameters.
      >>> # TODO: Assign the result to ridge. Fit the training data similar to linear regression algorithm.
      >>> ridge = Ridge()
      >>> ridge.fit(X_train, y_train)


      >>> # # 5) Make Predictions

      >>> # Make predictions on the testing set
      >>> lr_predictions = lr_model.predict(X_test)


      >>> # TODO: Make predictions on the random forest regression model and ridge model. Assign the predictions to `rf_predictions` and `ridge_predictions`
      >>> rf_predictions = rf_model.predict(X_test)
      >>> ridge_predictions = ridge.predict(X_test)


      >>> # # 6) Evaluate the Model

      >>> # Evaluate the performance of all the models
      >>> lr_mae = mean_absolute_error(y_test, lr_predictions)
      >>> lr_r2 = r2_score(y_test, lr_predictions)
      >>> lr_rme = root_mean_squared_error(y_test, lr_predictions)

      >>> #TODO: In the same fashion, fetch MAE (mean absolute error), R2 Score and RMSE (Root mean squared error) for the remaining two models.
      >>> rf_mae = mean_absolute_error(y_test, rf_predictions)
      >>> rf_r2 = r2_score(y_test, rf_predictions)
      >>> rf_rme = root_mean_squared_error(y_test, rf_predictions)

      >>> ri_mae = mean_absolute_error(y_test, ridge_predictions)
      >>> ri_r2 = r2_score(y_test, ridge_predictions)
      >>> ri_rme = root_mean_squared_error(y_test, ridge_predictions)


      >>> # # 7) Validate the Output

      >>> print("Linear Regression:")
      >>> print(f"Mean Absolute Error: {lr_mae}")
      >>> print(f"R-squared: {lr_r2}")
      >>> print(f"Root Mean Squared Error: {lr_rme}")

      >>> print("\nRandom Forest Regression:")
      >>> print(f"Mean Absolute Error: {rf_mae}")
      >>> print(f"R-squared: {rf_r2}")
      >>> print(f"Root Mean Squared Error: {rf_rme}")

      >>> print("\nRidge Regression:")
      >>> print(f"Mean Absolute Error: {ri_mae}")
      >>> print(f"R-squared: {ri_r2}")
      >>> print(f"Root Mean Squared Error: {ri_rme}")

          Linear Regression:
          Mean Absolute Error: 0.5332001304956557
          R-squared: 0.5757877060324508
          Root Mean Squared Error: 0.7455813830127764

          Random Forest Regression:
          Mean Absolute Error: 0.32754256845930246
          R-squared: 0.8051230593157366
          Root Mean Squared Error: 0.5053399773665033

          Ridge Regression:
          Mean Absolute Error: 0.5332039182571163
          R-squared: 0.5758549611440127
          Root Mean Squared Error: 0.7455222779992701
   --------------------------------------------

