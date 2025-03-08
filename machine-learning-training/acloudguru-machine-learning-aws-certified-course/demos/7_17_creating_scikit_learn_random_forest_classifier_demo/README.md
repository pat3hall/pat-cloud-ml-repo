------------------------------------------------------
7.17 Creating a scikit-learn Random Forest Classifier in Amazon SageMaker


  Resources

    Note: Downloaded github demo files to:
     C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\7_17_creating_scikit_learn_random_forest_classifier_demo

     -> Jupyter Notebook:
        CreateAScikit-LearnRandomForestClassifier.ipynb
        -> original notebook: need to rename without '.orig' to use else you may have a load error
           CreateAScikit-LearnRandomForestClassifier.orig.ipynb
     -> dataset data:
          data.csv
     -> generated:
        random forest decision tree graph: tree.png


   sklearn.metrics.roc_curve(y_true, y_score, ...)
     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
     - Compute Receiver operating characteristic (ROC).
     - Note: this implementation is restricted to the binary classification task.
     Returns:
    fpr: ndarray of shape (>2,)
        Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
    tpr: ndarray of shape (>2,)
        Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
    thresholds: ndarray of shape (n_thresholds,)
        Decreasing thresholds on the decision function used to compute fpr and tpr.

   sklearn.metrics.auc(x, y)
     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
     - Compute Area Under the Curve (AUC) using the trapezoidal rule.
     - This is a general function, given points on a curve.
     Parameters:
       x: array-like of shape (n,)
           X coordinates. These must be either monotonic increasing or monotonic decreasing.
       y: array-like of shape (n,)
           Y coordinates.
     Returns:
       auc: float
          Area Under the Curve.

  scikit-learn, a machine learning framework.
    - Designed to be native to Python, scikit-learn contains various classification, regression, and clustering algorithms,
      including random forests which we use in this lab.



About this lab

Scikit-learn is a great place to start working with machine learning. In this lab, we will use scikit-learn to create a
Random Forest Classifier to determine if you prefer cats or dogs. The data set being used is entirely made up, but could
easily be swapped with one of your own!

Learning objectives
  - Navigate to the Jupyter Notebook
  - Load and Prepare the Data
  - Train the Random Forest Model
  - Evaluate the Model
  - Predict for Yourself



  Creating an MXNet Image Classification in SageMaker Flow:

                             Survey
  SageMaker Notebook ------- Data  -------------------------->  Scikit Learn

                                                                  Model

     SageMaker                          Datasets                  Survey Questions
                          Training 80%           Testing 20%      1. Do you like walking?
                                                                  2. Do you like running?
                                                                  3. What is your favorite 'color'?
                                                                  4. What 'distance' do you walk?
                                                                  -> Do you prefer 'cats' or 'dogs'?
      ------------------------------------------------------
Solution
Log in to the Lab Environment

    To avoid issues with the lab, open a new Incognito or Private browser window to log in to the lab. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region.

Navigate to the Jupyter Notebook

    In the search bar, type "Sagemaker" to search for the Sagemaker service.
    Click on the Amazon SageMaker result to go directly to the Sagemaker service.
    Click on the Notebook Instances button to look at the notebook provided by the lab.
    Check to see if the notebook is marked as InService. If so, click on the Open Jupyter link under Actions.
    Click on the CreateAScikit-LearnRandomForestClassifier.ipnyb file.

Load and Prepare the Data

    Make sure you have the conda_python3 kernel.
        Check the bar in the upper right corner to see if it contains conda_python3.
        If it does not, click on Kernel in the menu, select Change kernel, and then select conda_python3 from the list. The version number available to you may differ.

    Under 1) Import Libraries, run the cell containing the import code by either clicking the Run button in the menu or pressing Ctrl + Enter to import the standard Python libraries as well as scikit-learn.

    Under 2) Prepare the Data, update and run the first cell to load the survey data from the data.csv file.

    df = pd.read_csv("data.csv")

    Under 2) Prepare the Data, run the second cell to look at the top ten results.

    Run the third cell to look at the data types for the different columns.

    Run the fourth cell to change the column names for more clarity.

    Run the fifth cell to describe the data as a whole and obtain statistical information about the data.

    Update and run the sixth cell to format the data so that the model will understand it better, specifically by changing some answer values into boolean data type or categorical data type.

    df['walk'] = df['walk'].astype('bool')
    df['run'] = df['run'].astype('bool')
    color_type = CategoricalDtype(categories=['red','green','blue'])
    df['color'] = df['color'].astype(color_type)
    df['label'] = df['label'].astype('bool')

    Run the seventh cell to look at the data types for the different columns.

    Update and run the eighth cell to perform a one hot encoding process to format the categorical data type.

    df = pd.get_dummies(df, prefix=['color'])

    Run the ninth cell to look at the top ten results again, now that the data has been reformatted.

    Update and run the tenth cell to split the data into training and testing sets, using 20% of the data for testing and 80% of the data for training.

    X_train, X_test, y_train, y_test = train_test_split(df.drop(labels='label', axis=1), df['label'], test_size = .2, random_state = 10)

Create the scikit-learn Model

    Under 3) Create the Model, run the first cell to create a Random Forest Classifier model using scikit-learn.

    Under 3) Create the Model, update and run the second cell to train the model using the training data.

    model.fit(X_train, y_train)

Evaluate the Model

    Under 4) Evaluate the Model, run the first cell to grab the estimator from the trained model and look at one of the decision trees.

    Under 4) Evaluate the Model, run the second cell to generate a graph of that decision tree.

    Update and run the third cell to run the testing data through the model.

    y_predict = model.predict(X_test)

    Run the fourth cell to generate a confusion matrix for the testing data and see how well the model performed.

    Run the fifth cell to format the confusion matrix using the seaborn library.

    Run the sixth cell to calculate the sensitivity and specificity from the confusion matrix.

    Run the seventh cell to plot the ROC curve.

    Run the eighth cell to calculate the area under the curve.

Predict for Yourself

    Under 5) Predict for yourself, type in answers for yourself in the first cell to create your own survey response.
    Under 5) Predict for yourself, run the first cell to have the model predict if you prefer cats or dogs.

Conclusion

Congratulations - you've completed this hands-on lab!

      ------------------------------------------------------

