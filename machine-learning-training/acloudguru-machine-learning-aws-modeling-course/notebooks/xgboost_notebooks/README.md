------------------------------------------------------
3.1 Reviewing SageMaker's XGBoost Algorithm

  XGBoost Algorithm.
    - a popular implementation of gradient boosted trees algorithm.
    - uses: ensemble learning, boosting, and gradient boosting
    Ensemble Learning
     - multiple ML models are combined to improve prediction accuracy as compared to a single ML model.
    Boosting
      - an ensemble learning technique that sequentially combines their predictions and improve the overall
        performance of the model.
      steps:
        - start by initializing the same weights to all the models.
        - The model is then trained against a subset of training data.
        - The error of the weak learner is computed.
        - Models with larger error rate are assigned higher weights compared to models that perform better,
          and they are retrained.
        - The previous two steps are repeated multiple times.
        - The final prediction is based on the weighted total of all the weak learners.
      Boosting Algorithm Types
        Adaptive BOosting
          - extensively used in classification problem
      Gradient BOosting
        - can be applied both to regression and classification problems.
        - uses gradient descent algorithm to minimize the errors.

    Decision tree algorithm
      - can be used to predict a category in a classification problem or a continuous numeric value in a
        regression problem.
      - divide the dataset into smaller subsets based on their features.
      - predicts the output by evaluating a sequence of if-then-else and true or false feature questions
        and estimating the minimum number of branches needed to assess the probability of making the right decision.

    Gradient Boosted Decision Tree
      - a decision tree ensemble learning algorithm that combines multiple machine learning model algorithms
         to obtain a better model.
      - XGBoost is an implementation of gradient boosted decision tree algorithm.

  SageMaker XGBoost Algorithm Attributes
      - both as a built-in algorithm or as a framework
    Learning Type:
      - both classification and regression
    File/Data Types:
      - libsvm, CSV, parquet, and protobuf.
    Instance Type:
      - CPU and GPU training process.
    Hyperparameters
      - required: num_round and num_class
      - 30+ optional hyperparameters to tune
      Num_round
        - the number of rounds to run the training
      num_class
        - number of classes
    Metrics:
      - reports MAE, MSE, RMSE, and MAP as metrics to measure a regression problem,
      - reports accuracy, area under curve (AUC), and F1 score to measure a classification problem.

      Metric Notes:
         MAE: Mean absolute error
            MAE(X,h) =  (1/m) SUM |(h(xi) - yi)|  where SUM is from i=1 to i=m
         MSE: Mean Square Error
            MSE(X,h) = [ (1/m) SUM (h(xi) - yi)**2 ] where SUM is from i=1 to i=m
         RMSE: Root Mean Square Error
            RMSE(X,h) = [ (1/m) SUM (h(xi) - yi)**2 ]**1/2 where SUM is from i=1 to i=m

              m: number of instances in the dataset
              xi: a vector of all the features values of the ith instances of the dataset, and
                  yi is the label (desired output)
              X: is a matrix containing all the features excluding labels of all instances in the dataset
              h: your systems prediction's function, also called the 'hypothesis'

         MAP: Mean Absolute Precision
             MAP = 1/m SUM AP_i    where SUM is from 1=1 to i=m; AP_i: average precision of instance i

         MAPE: Mean Absolute Percentage Error
            MAPE(X,h) =  (100/m) SUM |[(h(xi) - yi)/ yi]|  where SUM is from i=1 to i=m

  Business Use Cases
    finance domain
      - detect frauds
      - predict stock prices.
    e-commerce
      - predict customer churn
      - forecast sales.
    marketing
      - to predict ad-click revenue
      - customer segmentation.

  XGBoost algorithm with Amazon SageMaker AI
  https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html

  XGBoost sample notebooks
    https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-sample-notebooks.html
      Regression with Amazon SageMaker XGBoost (Parquet input)


  The following list contains a variety of sample Jupyter notebooks that address diffeerent use cases of Amazon SageMaker AI XGBoost algorithm.

    How to Create a Custom XGBoost container
    – This notebook shows you how to build a custom XGBoost Container with Amazon SageMaker AI Batch Transform.
      https://sagemaker-examples.readthedocs.io/en/latest/aws_sagemaker_studio/sagemaker_studio_image_build/xgboost_bring_your_own/Batch_Transform_BYO_XGB.html
         Save to: notebooks/xgboost_notebooks/Batch_Transform_BYO_XGB.ipynb

   Regression with XGBoost using Parquet
    – This notebook shows you how to use the Abalone dataset in Parquet to train a XGBoost model.
      https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_parquet_input_training.html
         Save to: notebooks/xgboost_notebooks/xgboost_parquet_input_training.ipynb

   How to Train and Host a Multiclass Classification Model
    – This notebook shows how to use the MNIST dataset to train and host a multiclass classification model.
         https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/xgboost_mnist/xgboost_mnist.html
         Save to: notebooks/xgboost_notebooks/xgboost_mnist.ipynb


