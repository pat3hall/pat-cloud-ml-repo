------------------------------------------------------
3.3 Exploring SageMaker's Linear Learner Algorithm

  Sagemaker linear learner algorithm.
    - A linear learner is a supervised learning algorithm that can be used to address classification or
      regression problems.
    - It can handle large data sets and high dimensional features.

  Linear learner
      - formula:   Y = m1*x1 + + m2*x2 + ...+ mn*xn + b

      - Training the linear learner model is like adjusting these weights and biases to minimize the difference
        between the predictor and actual home price.
      - uses a stochastic gradient descent to best fit the line to the data points.
      - It iteratively adjusts the model parameters to minimize the loss function.

  Sagemaker linear learner algorithm Attributes
    Learning Type:
      - classification and regression
    File/Data Types:
      -  CSV, protobuf, JSON (inference only)
    Instance Type:
      - CPU and GPU training process.
    Hyperparameters
      - required: num_class and predictor_type
      num_class
        - number of classes
      predictor_type
        - Specifies the type of target variable as a binary classification, multiclass classification, or regression.
    Metrics:
      - Cross entropy loss, absolute error, and MSE (regression metrics)
      - Precision, Recall, accuracy (classification metrics)

  Amazon SageMaker -> Developer Guide -> Linear learner hyperparameters
    https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html
    num_classes
      - The number of classes for the response variable. The algorithm assumes that classes are labeled 0, ...,
        num_classes - 1.
      - Required when predictor_type is multiclass_classifier. Otherwise, the algorithm ignores it.
      - Valid values: Integers from 3 to 1,000,000
    predictor_type
      - Specifies the type of target variable as a binary classification, multiclass classification, or regression.
      - Required
      - Valid values: binary_classifier, multiclass_classifier, or regressor


  Linear Learner business use cases
     processing loan applications
       - based on financial history,
     detecting email spam
       - based on content and sender information,
    recommendation systems
      - to create personalized recommendations for product, music and movies.


   Linear Learner Algorithm
   https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html

      An Introduction with the MNIST dataset

        Using the MNIST dataset, we train a binary classifier to predict a single digit.
          https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist.html
             saved to: notebooks/linear_learner_notebooks/linear_learner_mnist.ipynb

