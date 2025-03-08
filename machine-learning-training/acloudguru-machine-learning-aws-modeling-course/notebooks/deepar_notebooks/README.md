------------------------------------------------------
3.6 Discovering SageMaker's DeepAR Forecasting Algorithm


  SageMakers's DeepAR algorithm
    - It is a supervised learning algorithm designed for forecasting one-dimensional time-series data
      using a recurrent neural network.
    one-dimensional time-series data
      - sequential data where each observation corresponds to a single variable measured at regular time intervals.
      - For example, daily temperature measurements and hourly stock prices.

  DeepAR algorithm - Analogy
    - a shoe company getting ready to introduce a new shoe design, and wants to predict how many they
      expect to sell
    cold start problem.
      - means that we don't have enough history to pour into the forecasting model and deliver how many we
        expect to sell.
    - want to create a model that allows us to combine those patterns of multiple shoes to create an aggregate
      forecast that's probably a more realistic model for our brand new shoe.
    - can use DeepAR to look at multiple datasets of historic data for multiple shoes.

  DeepAR types of forecasts:
    point-in-time forecast
      - provides a single predicted value for each time step in the forecast period.
      - an example of that is just the number of sneakers that we might sell in a week is X.
    probabilistic forecast,
      - provides a range of possible future values for each time step along with associated probabilities.
      - an example of this might be the number of sneakers sold in a week is between X and Y with Z probability.

  DeepAR input file format:
    - Each observation in the input file should contain the following fields:
      Start
         - required field; a string representation of a timestamp
         - A string with the format YYYY-MM-DD HH:MM:SS
         - it cannot contain time zone information.
      target
          - required field; an array of floating-point values or integers that represents the time-series data.
      dynamic_feat field
        - optional;  an array of arrays of floating-point values or integers that represents the vector
          of custom feature time-series data.
        - For example, if the time-series data represents the stock prices, an associated dynamic_feat might be a
          Boolean condition that indicates if it's a favorable economic condition or not.
      cat
        - optional; an array of categorical features that can be used to encode the groups that the record belongs to.
        - if you use this field, all time-series must have the same number of categorical features.

  deepAR file format info:
    Use the SageMaker AI DeepAR forecasting algorithm
      https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html

  Sagemaker DeepAR algorithm Attributes
    Learning Type:
      - Forecasting (time-series)
    File/Data Types:
      - training and testing dataset can be provided in JSON Lines format.
      - input files can be in gzip and parquet format.
      - JSON for inferencing
    Instance Type:
      - train on CPU and GPU in both single and multi-machine settings
      - recommended starting with CPU [ex: ml.c4.2xlarge, ml.c4.4xlarge] and switch to GPU on when necessary
    Hyperparameters
      - required: context_length, epochs, prediction_length, & time_freq
      context_length
        - specifies the number of time points that the model gets to see before making the prediction.
      epochs
        - a maximum number of passes over the training data.
      prediction_length
        - the number of time steps that the model is trained to predict.
      time_freq
        - the granularity of the time-series in the dataset.
        - It can be monthly, weekly, daily, or hourly.
    Metrics:
      https://docs.aws.amazon.com/sagemaker/latest/dg/deepar-tuning.html
      test:RMSE
        - The root mean square error between the forecast and the actual target computed on the test set.
         - Minimize
      test:mean_wQuantileLoss
        - The average overall quantile losses computed on the test set.
        - To control which quantiles are used, set the test_quantiles hyperparameter.
        - Minimize
      train:final_loss
        - The training negative log-likelihood loss averaged over the last training epoch for the model.
        - Minimize


  DeepAR Business Use Cases:
    forecasting.
      demand forecasting
          - to forecast demand for products and services.
      Sales forecasting
         - to forecast sales volumes of a product for a time period,
      financial forecasting
        - to forecast market trends and exchange rates.
    Risk assessment and mitigation
      - forecasting risks in credit defaults or fraud incidents,


  Use the SageMaker AI DeepAR forecasting algorithm
    https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html

      DeepAR Sample Notebooks
        https://docs.aws.amazon.com/sagemaker/latest/dg/deepar-tuning.html
        Save as:
          notebooks/deepar_notebooks/DeepAR-Electricity.ipynb


