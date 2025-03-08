Amazon SageMaker’s Built-in Algorithm Webinar Series: DeepAR Forecasting
  https://www.youtube.com/watch?v=g8UYGh0tlK0

  Summary:
    - In this webinar, Kris Skrinak, AWS Partner Solution Architect, will deep dive into time series 
      forecasting with deep neural networks using Amazon SageMaker built-in algorithm: DeepAR Forecasting.


  How AWS Chooses its Built-in Algorithms
     - algorithm needs to popular
     - seek to provide 10x+ performance improvement 
       - how to get 10X performance improvement
         - stream whenever possible
         - seek algorithms that see data once and only once
         - create a shared state especially when we branch out to training clusters
         - seek to use GPUs whenever possible 
           - note: K-means, LDA (Linear Discriminant Analsis), XGBoost are CPU only algorithms
         - seek algorithms that are distributed - that run efficiently in distributed environments
         - use abstraction and containerization
         - strongly faver record i/o protobuf for file format
            - reduces data size for JPEG
            - allows for continuous reading

  4 Ways to use Sagemaker
    - console
    - CLI
    - SPARC
    - notebooks

  Time Series Forecasting
    Use Cases
      - supply chain efficiencies by doing demand predictions 
      - avoiding outages
      - allocating computing resources more effectively by predicting webserver traffic
      - saving lives by forecasting patients visits and staffing hospitals
      - predictions for urban planning (i.e. housing prices, traffic)
      - Amazon uses it for product demand in our fulfillment centers particular for key dates (prime day, black friday)
     algorithms
      - linear regression remains a common method for forecasting, but we have now entered the era 
        of deep linear networks
      - SageMaker DeepAR algorithm is specifically designed for forecasting

    Definitions
      Recurrent Neural Net (RNN)
        - type of deep learning networks where connections between nodes form a directed graph along a
          sequence. This allows them to exhibit temporal dynamic behavior for a time sequence so 
        - unlike feed-forward neural networks, RNNs can use their internal state to process sequences of
          inputs an 
        - long Short-Term Memory (LSTM): 
          - type of RNN
          - can process data sequentially and keep its state hidden through time a common
          - a LSTM is commonly composed of a cell, an input, an output, and a forget gate 
          - the 'cell' remembers values over an arbitrary time intervals, and the other 3 gates regulate 
            the flow of information into and out of the cell
      Root Mean Square Error (RMSE)

      One-step-ahead forecasts
          - when predicting for multiple data points, one step ahead forecast update the history with
            the correct known value - gives insite into how our model is performing, but also can present
            an optimistic evaluation of the forecast

      Horizontal forecasts
       - in horizon models, when forecasing out-of-sample, each forecast builds off of the forecasted periods
         that precede, so errors early on in test data can't compound to create large deviations late in the
         test data, 
       - although this can be more realistic, it can be difficult to create the forecasts particularly
         as the model complexity increases

      Hold-out test set

      Cold start forecasting
        - cold start scenario occurs when want to generate a forecast for a time series with little or no
          historical data
        - occurs frequently in practices such as in scenarios when new products are introduced or 

      Probabilistic forecasts
        - probabilistic forecasts are used in DeepAR and they produce both point forecasts and probalistics
          forecasts
        - for example, the amount of sneakers sold in a a week is between x and y with z probability
        - well-suited for business applicatiosn such as capacity planning where specific forecast quantities
          are more important than the most likely outcome


    Stationary Series:
      - a stochastic process whose unconditional joint probability distribution does not change when shifted
        in time. 
      - Consequently, parameters such as mean and variance also do not change over time
      - stationary Series example: 
          - modulization with-in a fixed range 
      - non-stationary Series examples: 
         - modulization that is rising over time
         - modulization varires greatly in cadence or periodicity

    Time Series dataset requirements
      - need to convert data to stationary series of data??

    Traditional forecasting
       Time Series Forecasting with Linear Learner
         uses "linear_time_series_forecast.ipynb"
           -> previously provided in SageMaker Examples, but I could not find it
           - uses gasoline consumption volumes dataset to predict trends
           - for time series dataset, you need to use the end of your dataset for validation/testing
             (instead of randomly selected data)

   Forecasting with Deep Neural Networks
      What problem are we solving?
        - Magnitudes of time series differ widely
        - distribution of magnitudes are strongly skewed

      Solution
        - see Amazon White paper on DeepAR
        - Employ an RNN deep learning architecture for probabilistic forecasting
          - incorporating a negative binomial likihood for count data
          - special treatment for the case where the magnitudes vary widely

      Dataset Requirements
        Must be a stationary Dataset
          - in case you get a non-stationary series, you need to stationarize the series: detrending, 
            differencing, seasonality

        Gzipped JSON lines or Parquet file

        Format:
          - start: a string of the format YYY-MM-DD HH:MM:SS
          - target: an array of floats (or integers) that represent the time series
          - dynamic_feat (optional)
          - cat (optional): an array of categorical features
            - one of the principal reasons for using DeepAR is because you have a very large set of categorical
              data that relates to each other

          ["start": "2009-11-01 00:00:00", "target": [4.3, "NaN", 5.1, ...], "cat": [0, 1], "dynamic_feat": [[1.1, 1.2, 0.5, ...]]]
          ["start": "2012-01-30 00:00:00", "target": [1.0, -5.0, ...], "cat": [2, 3], "dynamic_feat": [[1.1, 2.05, ...]]]
          ["start": "1999-01-30 00:00:00", "target": [2.0, 1.0], "cat": [1, 4], "dynamic_feat": [[1.3, 0.4]]]


     DeepAR meaning
       - 'AR' stands for 'Auto-Regressive' model
       - AR model builds into the fact that spiky model but with longer tail (slower or less spikey drop off)
       Auto-Regressive model
         x(t) = alpha * x(t - 1) + error(t)
         - spiky model but does not include that regressive drop off
       Moving Average Model
         x(t) = beta * error(t - 1) + error(t)

     DeepAR Advantages
       - minimal manual feature engineering needed to capture complex group-dependent behavior
          - because the deep neural models are doing some of that feature engineering for you
       - probabilistic forecasts in the form of Monte Carlo samples can be used to compute consistent quantile estimates
         for all sub-ranges in the prediction horizon
       - by learning from similar items, the method is able to provide forecasts for items with little or no history at all

       - support for different types of time series: real numbers, counts, and values in an interval

       - automatic evaluation of model accuracy in a backtest after training
       - Engineered to use either GPU or CPU hardware to train its long short-term memory (LSTM) based on
         RNN model quickly and flexibly
       - Scales up to datasets comprising 100,000+ time series
       - suport for training data in JSON Lines or Parquet format
       - supports missing values, categorical and time series features, and generalized frequencies


     DeepAR Architecture
        - includes both Seq2Seq algorithm and LSTM algorithm
        - using encoder / decoder model for our predictions

          Model (see papers at archive.org (?))
             Training                         Prediction
             Encoder        -- Seq2Seq ->     Decoder
          (includes LSTM)

     Forecasting Accuracy
       - if a large dataset with a large number of related time series, results are exceptional

     DeepAR Hyperparameters

        time_freq
          - The granularity of the time series in the dataset. 
          - Use time_freq to select appropriate date features and lags. 
          - The model supports the following basic frequencies. It also supports multiples of these basic frequencies. 
            For example, 5min specifies a frequency of 5 minutes.
            M: monthly;   W: weekly;   D: daily;   H: hourly;   min: every minute

        Context_length
          - The number of time-points that the model gets to see before making the prediction. 
          - The value for this parameter should be about the same as the prediction_length

        prediction_length
          - The number of time-steps that the model is trained to predict, also called the forecast horizon. 
          - The trained model always generates forecasts with this length. It can't generate longer forecasts. 
          - The prediction_length is fixed when a model is trained and it cannot be changed later.

        num_cells
          - The number of cells to use in each hidden layer of the RNN. Typical values range from 30 to 100.
          - Optional; Valid values: positive integer; Default value: 40

        num_layers
          - The number of hidden layers in the RNN. Typical values range from 1 to 4.
          - Optional;  Valid values: positive integer;  Default value: 2

        likelihood
           - The model generates a probabilistic forecast, and can provide quantiles of the distribution and return samples. 
           - Depending on your data, select an appropriate likelihood (noise model) that is used for uncertainty estimates. 
           - The following likelihoods can be selected:
             gaussian: 
               - Use for real-valued data.
             beta: 
               - Use for real-valued targets between 0 and 1 inclusive.
             negative-binomial: 
               - Use for count data (non-negative integers).
             student-T: 
               - An alternative for real-valued data that works well for bursty data.
             deterministic-L1: 
               - A loss function that does not estimate uncertainty and only learns a point forecast.
           - Default value: student-T

        epochs
          - The maximum number of passes over the training data. 
          - The optimal value depends on your data size and learning rate. 
          - Typical values range from 10 to 1000.

        mini_batch_size
          - The size of mini-batches used during training. Typical values range from 32 to 512.
          - Optional;  Valid values: positive integer;  Default value: 128

        learning_rate
          - The learning rate used in training. Typical values range from 1e-4 to 1e-1. (default: 1e-3)

        dropout_rate
          - The dropout rate to use during training. 
          - The model uses zoneout regularization (to improve regularization and to prevent overfitting). 
          - For each iteration, a random subset of hidden neurons are not updated. 
          - Typical values are less than 0.2.
          - Optional; Valid values: float; Default value: 0.1

        early_stopping_patience
          - If this parameter is set, training stops when no progress is made within the specified number of epochs. 
          - The model that has the lowest loss is returned as the final model.
          - Optional; Valid values: integer


   Code: Time series with forecasting with DeepAR (deepar_synthetic.ipynb -  no longer exists)

import time
import numpy as np
import pandas as pd
np.random.seed(1)
import matplotlib.pyplot as plt
import json

# will use the sagemaker client library for easy interface with sagemaker and
#  s3fs for uploading the training data to S3
!conda install -y s3fs
import boto3
import s3fs
import sagemaker
from sagemaker import get_execution_role


bucket = 'pat-demo-bkt-e2'
prefix = "sagemaker/DEMO-deepar"  # prefix used for all data stored within the bucket

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()  # IAM role to use by SageMaker

region = sagemaker_session.boto_region_name

s3_data_path = "s3://{}/{}/data".format(bucket, prefix)
s3_output_path = "s3://{}/{}/output".format(bucket, prefix)



# Generating and uploading data
# In this toy example, we want to train a model that can predict the next 48 points on synthetically
#    generated time series.. The time series that we use hourly granularity

freq = 'H'
prediction_length = 48

# also need to configure the so-called 'context-length', which determines how much context of the time
#   series the model should take into account when making the prediction, i.e. how many previous points
#   to look at. A typical value to start with is around the same size as the 'prediction_length'. IN our
#   example, we will use a long 'context_length' of 72. Not that in addition to the 'context_length' the model
#   also takes into account hte values of the time series at typical seasonal windows e.g. for hourly data
#   The model will look at the value of the series 24h ago, one week ago, one month ago, etc. So it is
#   not necessary to make the 'context_length' span an entire month if you expect month seasonality in your
#   hourly data

context_length = 72

# for this notebook, we will generate 200 noisy time series, each consisting of 400 data points an with 
#   seasonality of 24 hours. In our ddummy example, al time series start at the same time point t0. When
#   preparing your  data, it is important to the correct start point for each time series, because the model 
#   uses the time-point as a frame of reference, which enables it to learn e.g. that weekdays behave differenctly
#   from weekends

t0 = '2016-01-01 00:00:00'
data_length = 400
num_ts = 200
period = 24

# each time series will be a noisy sine wave with a random level

time_series = []
for k in range (num_ts):
    level = 10 * np.random.rand()
    seas_amplitude = (0.1 + 0.3 * np.random.rand()) * level
    sig = 0.5 * level # noise parametet (constant in time)
    time_ticks - np.array(range(data_length))
    source = level + seas_amplitude * np.sin(time_ticks * (2 * np.pi)/period)
    noise = sig * np.random.randn(data_length)
    data = source + noise
    index = pd.DatetimeIndex(start=t0, freq=freq, periods=data_length)
    time_series.append(pd.Series(data=data, index=index))

   #  pd.DateimeIndex -> fails because it no longer has 'start' parameter
   TypeError                                 Traceback (most recent call last)
   Cell In[12], line 10
         8 noise = sig * np.random.randn(data_length)
         9 data = source + noise
   ---> 10 index = pd.DatetimeIndex(start=t0, freq=freq, periods=data_length)
        11 time_series.append(pd.Series(data=data, index=index))
   
   TypeError: DatetimeIndex.__new__() got an unexpected keyword argument 'start'

plt.rcParam['figure.figsize'] = [20,5]
time_series[0].plot()


# often one is interested in tuning or evaluating the model by look at error metrics on a hold-out set. 
# For other machine learning tasks such as classification, one typically does this by randomly separating examples
# into train/test sets. For forecasting it is import to do this train/test split in time rather that by series


# in this example, we will leave out the last section of each time series we just generate and use only the first
# part as training data. Here we will predict 48 data points, therefore we take out the trailing 48 points 
# from each time series to define the training set. The test set contains the full range of each time series


time_series_training = []
for ts in time_series:
    time_series_training.append(ts[:-prediction_length])

time_series[0].plot(label='test')
time_series_training[0].plot(label='train', ls=':')
plt.legend()
plt.show()

# the following utility functions convert pandas.Series objects into the appropriate JSON strings that DeepARM can consume.
# We will use these to writ the data to S3.

def series_to_obj(ts, cat=None):
   obj = {"start": str(ts.index[0]), "target": list(ts)}
   if cat is not None:
      obj["cat"] = cat
   return obj

def series_to_jsonline(ts, cat=None):
    return json.dumps(series_to_obj(ts, cat))


encoding = 'utf-8'
s3filesystem = s3fs.S3FileSystem()

with s3filesystem.open(s3_data_path + "/train/train.json", 'wb') as fp:
    for ts in time_series_training:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))

# Train a model


# We can non define the estimagor that will launch the training job


# Note: get_image_uri -> in SageMaker 2, changed to: sagemaker.image_uris.retrieve():
# from sagemaker.amazon.amazon_estimator import get_image_uri
# image_name = get_image_uri(region,  'forecasting-deepar')

from sagemaker import image_uris

image_name = image_uris.retrieve(region=region, framework="forecasting-deepar")

deepar = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_name=image_name,
    role=role,
    instance_count=1,
    instance_type="ml.c4.xlarge",
    base_job_name='DEMO-deepar',
    max_run=1800,  # max training time in seconds
    output_path="s3://{}/{}".format(bucket, s3_output_path),
)



# Next, we need to set some hyperparmaters: for example: frequency of the time series used, number of data points
# the model will look at in the past, number of predicted data points. The other hyperparameters concern the model 
# to train (number of layers, number of cells per layer, likelihood function) and the training options such as number
# epochs, batch size, learning rate.

hyperparameter = {
    "time_freq": freq,
    "context_length": str(context_length),
    "prediction_length": str(prediction_length),
    "num_cells": "40",
    "num_layers": "3",
    "likelihood": "gaussian",
    "epochs": "20",
    "mini_batch_size": "32",
    "learning_rate": "0.001",
    "dropout_rate": "0.05",
    "early_stopping_patience": "10"
}

estimator.set_hyperparameters(**hyperparameters) 

# we are ready to launch the training job. Sagemaker will start an EC2 instance, download the data from S3,
# start training the mode and save the trained model

# if you provide the 'test' data channel, as we do in this example, DeepAR will also calculate the accuracy 
#  metrics for the trained model on this test data set. This is done by predicting the last 'prediction_length' points
#  of each teim series in the test set and comparing this to the actual value of the time series. The computed error
#  metrics will be included as part of the log output

# Note: the next cell may take a few minutes to complete...

data_channels = {
    "train": "s3://{}/train/".format(s3_data_path)
    "test": "s3://{}/test/".format(s3_data_path)

estimator.fit(inputs=data_channels)

....
billable seconds: 167


# Create endpoint and predictor

# Now that we have trained a model, we cvan use it to perform predictions by deploying it to an endpoint.

# Note: remember to delete the endpoint after running this experiment

# job_name = estimator.lastest_training_job.name

# endpoint_name - sagemaker_session.endpoint_from_job(job_name=job_name, initial_instance_count=1, instance_type='ml.m4.xlarge',
# deployment_image=image_name, role=role)

# to query the endpoint and perform predictions, we can define the following utility class: this allows making
# requests using pandas.Series object rather than raw JSON strings

Class DeepARPredictor(sagemaker.predictor, RealtTimePredictor):
    def set_prediction_parameters(self, freq, prediction_length):
        """Set the time frequency and prediction length parmameters. This method **must** be called
        befor being able to use 'predict'

        Parameters:
        freq -- string indicating the time frequency
        prediction_length -- integer, number of predicted time points

        Return values: none.
        """
        self.freq=freq
        self.prediction_length = prediction_length

    def predict(self, ts, cat=None, encoding="utf-8", num_samples=100, quantiles=["0.1", "0.5", "0.9"])
        """Requests the prediction of for the time series listed in `ts`, each with the (optimal)
        corresponding category listed in `cat`

        Parameters:
        ts -- list of `pandas.Series` objhect, the time series to predict
        cat -- list of integers (default: None)
        encoding -- string, encode to use for the request (default: "utf-8")
        num_samples -- integer, number of samples to compute at prediction time (default: 100)
        quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"0]

        Return value: list of `pandas.DataFrame` objects, each containing the predictions
        """
        prediction_times = [x.index[-1]+1 for x in ts]
        req = self.__encode_request(ts, cat, encoding, num_samples, quantiles)
        req = super(DeepARPredictor, self).predict(req)
        return self.__decode_reponse(res, predition_times, encoding)


    def __encode_request(self, ts, cat, encoding, num_samples, quantiles):
        instances = [series_to_obj(ts[k], cat[k] if cat else None) for k in range(len(ts))]
        configuration = {"num_samples": num_samples, "output_types: ["quantiles"], "quantiles": quantiles}
        http_request_data = {"instances": instances, "configuration": configuration}
        return json.dumps(http_request_data).encode(encoding)

    def __decode_response(self, response, prediction_times, encoding)
        reponse_data = json.loads(response.decode(encoding))
        list_of_df = []
        for k in range (leng(prediction_times)):
            prediction_index = pd.DateTimeIndex(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            list_of_df.append(pd.DataFrame(data=response_data['predictions'][k]['quantiles'], index=prediction_index)
        return list_of_df


endpoint_name="DEMO-deepar-2018-08-22-19-03-18-002"

predictor = DeepARPredictor(
    endpoint=endpoint_name,
    sagemaker_session=sagemaker_session,
    content_type="application/json"
)
predictor.set_prediction_parameters(freq, prediction_length)

# Make predictions and plot results

# Now we can use the previously created 'predictor' object. For simplicity, we will only predict the first few
# time series used for training, and compare the results with the actual data we kept in the test set.

. . .

# delete endpoint
sagemaker_session.delete_endpoint(endpoint_name)



   Code: SageMaker/DeepAR demo on electricity dataset


        >>> # # SageMaker/DeepAR demo on electricity dataset

        >>> # This notebook's CI test result for us-west-2 is as follows. CI test results in other regions can be found at the end of the notebook. 
        >>>  
        >>>  
        >>> # This notebook complements the [DeepAR introduction notebook]
        >>> #  (https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/deepar_synthetic/deepar_synthetic.ipynb). 
        >>>  
        >>> # Here, we will consider a real use case and show how to use DeepAR on SageMaker for predicting energy consumption of 370 customers over 
        >>> # time, based on a [dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) that was used in the academic papers 
        >>> # [[1](https://media.nips.cc/nipsbooks/nipspapers/paper_files/nips29/reviews/526.html)] and [[2](https://arxiv.org/abs/1704.04110)].  
        >>> # 
        >>> # In particular, we will see how to:
        >>> # * Prepare the dataset
        >>> # * Use the SageMaker Python SDK to train a DeepAR model and deploy it
        >>> # * Make requests to the deployed model to obtain forecasts interactively
        >>> # * Illustrate advanced features of DeepAR: missing values, additional time features, non-regular frequencies and 
        >>> #   category information
        >>> # 
        >>> # Running this notebook takes around 40 min on a ml.c4.2xlarge for the training, and inference is done on a ml.m5.large 
        >>> #   (the usage time will  depend on how long you leave your served model running).
        >>> # 
        >>> # This notebook is tested using SageMaker Studio but using classic Notebook (From the SageMaker Menu, go to Help -> select 
        >>> #  `Launch Classic Notebook`). 
        >>> # 
        >>> # For more information see the DeepAR [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html) or 
        >>> #  [paper](https://arxiv.org/abs/1704.04110), 

        >>> # install s3fs
        >>> # S3Fs is a Pythonic file interface to S3. It builds on top of botocore.
        >>> #  https://s3fs.readthedocs.io/en/latest/
        >>> %pip install --quiet --upgrade --upgrade-strategy only-if-needed s3fs

        >>> # if python 2, upgrade print() function 
        >>> # from __future__ import print_function

        >>> %matplotlib inline

        >>> import sys
        >>> import zipfile
        >>> from dateutil.parser import parse
        >>> import json
        >>> from random import shuffle
        >>> import random
        >>> import datetime
        >>> import os

        >>> import boto3
        >>> import s3fs
        >>> import sagemaker
        >>> import numpy as np
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from datetime import timedelta

        >>> from ipywidgets import interact, interactive, fixed, interact_manual
        >>> import ipywidgets as widgets
        >>> from ipywidgets import IntSlider, FloatSlider, Checkbox


        >>> # set random seeds for reproducibility
        >>> np.random.seed(42)
        >>> random.seed(42)


        >>> sagemaker_session = sagemaker.Session()


        >>> # Before starting, we can override the default values for the following:
        >>> # - The S3 bucket and prefix that you want to use for training and model data. This should be within the same region 
        >>> #   as the Notebook Instance, training, and hosting.
        >>> # - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.

        >>> s3_bucket = sagemaker.Session().default_bucket()  # replace with an existing bucket if needed
        >>> s3_prefix = "deepar-electricity-demo-notebook"  # prefix used for all data stored within the bucket

        >>> role = sagemaker.get_execution_role()  # IAM role to use by SageMaker


        >>> region = sagemaker_session.boto_region_name

        >>> s3_data_path = "s3://{}/{}/data".format(s3_bucket, s3_prefix)
        >>> s3_output_path = "s3://{}/{}/output".format(s3_bucket, s3_prefix)


        >>> # Next, we configure the container image to be used for the region that we are running in.

        >>> image_name = sagemaker.image_uris.retrieve("forecasting-deepar", region)


        >>> # ### Import electricity dataset and upload it to S3 to make it available for Sagemaker

        >>> # As a first step, we need to download the original data set of from the UCI data set repository.

        >>> DATA_HOST = f"sagemaker-example-files-prod-{region}"
        >>> DATA_PATH = "datasets/timeseries/uci_electricity/"
        >>> ARCHIVE_NAME = "LD2011_2014.txt.zip"
        >>> FILE_NAME = ARCHIVE_NAME[:-4]


        >>> s3_client = boto3.client("s3")

        >>> if not os.path.isfile(FILE_NAME):
        >>>     print("downloading dataset (258MB), can take a few minutes depending on your connection")
        >>>     s3_client.download_file(DATA_HOST, DATA_PATH + ARCHIVE_NAME, ARCHIVE_NAME)

        >>>     print("\nextracting data archive")
        >>>     zip_ref = zipfile.ZipFile(ARCHIVE_NAME, "r")
        >>>     zip_ref.extractall("./")
        >>>     zip_ref.close()
        >>> else:
        >>>     print("File found skipping download")


        >>> # Then, we load and parse the dataset and convert it to a collection of Pandas time series, which makes common time 
        >>> # series operations such as indexing by time periods or resampling much easier. The data is originally recorded in 15min 
        >>> # interval, which we could use directly. Here we want to forecast longer periods (one week) and resample the data to a 
        >>> # granularity of 2 hours.

        >>> # read_csv: separator: ';', column -> index (contains data/time -> parse index, decimal represented by ','
        >>> data = pd.read_csv(FILE_NAME, sep=";", index_col=0, parse_dates=True, decimal=",")
        >>> # each row is 1 time period -> 370 rows
        >>> num_timeseries = data.shape[1]
        >>> # convert eight 15 min rows to one 2 hour row
        >>> data_kw = data.resample("2H").sum() / 8
        >>> timeseries = []
        >>> for i in range(num_timeseries):
        >>>     # create timeseries with all zeros rows are (f=front) trimmed from 'data_kw' 
        >>>     timeseries.append(np.trim_zeros(data_kw.iloc[:, i], trim="f"))

        >>> data.shape
              (140256, 370)
        >>> data_kw.shape
              (17533, 370)
        >>> print(len(timeseries))
              370
        >>> print(len(timeseries[0]))
              13153

        >>> # Let us plot the resulting time series for the first ten customers for the time period spanning the first two weeks of 2014.

        >>> fig, axs = plt.subplots(5, 2, figsize=(20, 20), sharex=True)
        >>> axx = axs.ravel()
        >>> for i in range(0, 10):
        >>>     timeseries[i].loc["2014-01-01":"2014-01-14"].plot(ax=axx[i])
        >>>     axx[i].set_xlabel("date")
        >>>     axx[i].set_ylabel("kW consumption")
        >>>     axx[i].grid(which="minor", axis="x")


        >>> # ### Train and Test splits
        >>> # 
        >>> # Often times one is interested in evaluating the model or tuning its hyperparameters by looking at error metrics on a 
        >>> # hold-out test set. Here we split the available data into train and test sets for evaluating the trained model. 
        >>> # For standard machine learning tasks such as classification and regression, one typically obtains this split by randomly 
        >>> # separating examples into train and test sets. 
        >>> # However, in forecasting it is important to do this train/test split based on time rather than by time series.
        >>> # 
        >>> # In this example, we will reserve the last section of each of the time series for evalutation purpose and use only the first part 
        >>> # as training data. 

        >>> # we use 2 hour frequency for the time series
        >>> freq = "2H"

        >>> # we predict for 7 days
        >>> prediction_length = 7 * 12

        >>> # we also use 7 days as context length, this is the number of state updates accomplished before making predictions
        >>> context_length = 7 * 12


        >>> # We specify here the portion of the data that is used for training: the model sees data from 2014-01-01 to 2014-09-01 for
        >>> # training, dataset is for the year of 2014, and the train set is the 1st 8 months of 2014 

        >>> start_dataset = pd.Timestamp("2014-01-01 00:00:00", freq=freq)
        >>> end_training = pd.Timestamp("2014-09-01 00:00:00", freq=freq)


        >>> # The DeepAR JSON input format represents each time series as a JSON object. In the simplest case each time series just consists 
        >>> # of a start time stamp (``start``) and a list of values (``target``). For more complex cases, DeepAR also supports the fields 
        >>> # ``dynamic_feat`` for time-series features and ``cat`` for categorical features, which we will use  later.

            # convert timeseries list to a list of dict's containing 'start' date/time, 'target': [kw list]
        >>> training_data = [
        >>>     {
        >>>         "start": str(start_dataset),
        >>>         "target": ts[
        >>>             start_dataset : end_training - timedelta(days=1)
        >>>         ].tolist(),  # We use -1, because pandas indexing includes the upper bound
        >>>     }
        >>>     for ts in timeseries
        >>> ]
        >>> print(len(training_data))
            370


        >>> # As test data, we will consider time series extending beyond the training range: these will be used for computing test scores, 
        >>> # by using the trained model to forecast their trailing 7 days, and comparing predictions with actual values.
        >>> # To evaluate our model performance on more than one week, we generate test data that extends to 1, 2, 3, 4 weeks beyond the 
        >>> # training range. This way we perform *rolling evaluation* of our model.

        >>> num_test_windows = 4

            # convert timeseries list to a list of dict's containing 'start' date/time, 'target': [kw list]
            # where for each training dict, the test data contains 4 dicts  with 1, 2, 3, 4 weeks appended to the training data
        >>> test_data = [
        >>>     {
        >>>         "start": str(start_dataset),
        >>>         "target": ts[start_dataset : end_training + timedelta(days=k * prediction_length)].tolist(),
        >>>     }
        >>>     for k in range(1, num_test_windows + 1)
        >>>     for ts in timeseries
        >>> ]
        >>> print(len(test_data))


        >>> # Let's now write the dictionary to the `jsonlines` file format that DeepAR understands (it also supports gzipped jsonlines 
        >>> # and parquet).

        >>> def write_dicts_to_file(path, data):
        >>>     with open(path, "wb") as fp:
        >>>         for d in data:
        >>>             fp.write(json.dumps(d).encode("utf-8"))
        >>>             fp.write("\n".encode("utf-8"))


        >>> get_ipython().run_cell_magic('time', '', 'write_dicts_to_file("train.json", training_data)\nwrite_dicts_to_file("test.json", test_data)\n')


        >>> # Now that we have the data files locally, let us copy them to S3 where DeepAR can access them. Depending on your connection, this may take a couple of minutes.

        >>> s3 = boto3.resource("s3")


        >>> def copy_to_s3(local_file, s3_path, override=False):
        >>>     assert s3_path.startswith("s3://")
        >>>     split = s3_path.split("/")
        >>>     bucket = split[2]
        >>>     path = "/".join(split[3:])
        >>>     buk = s3.Bucket(bucket)

        >>>     if len(list(buk.objects.filter(Prefix=path))) > 0:
        >>>         if not override:
        >>>             print(
        >>>                 "File s3://{}/{} already exists.\nSet override to upload anyway.\n".format(
        >>>                     s3_bucket, s3_path
        >>>                 )
        >>>             )
        >>>             return
        >>>         else:
        >>>             print("Overwriting existing file")
        >>>     with open(local_file, "rb") as data:
        >>>         print("Uploading file to {}".format(s3_path))
        >>>         buk.put_object(Key=path, Body=data)


        >>> get_ipython().run_cell_magic('time', '', 'copy_to_s3("train.json", s3_data_path + "/train/train.json")\ncopy_to_s3("test.json", s3_data_path + "/test/test.json")\n')


        >>> # Let's have a look to what we just wrote to S3.

        >>> s3_sample = s3.Object(s3_bucket, s3_prefix + "/data/train/train.json").get()["Body"].read()
        >>> StringVariable = s3_sample.decode("UTF-8", "ignore")
        >>> lines = StringVariable.split("\n")
        >>> print(lines[0][:100] + "...")


        >>> # We are all set with our dataset processing, we can now call DeepAR to train a model and generate predictions.

        >>> # ### Train a model
        >>> # 
        >>> # Here we define the estimator that will launch the training job.

        >>> estimator = sagemaker.estimator.Estimator(
        >>>     image_uri=image_name,
        >>>     sagemaker_session=sagemaker_session,
        >>>     role=role,
        >>>     instance_count=1,
        >>>     instance_type="ml.c4.2xlarge",
        >>>     base_job_name="deepar-electricity-demo",
        >>>     output_path=s3_output_path,
        >>> )


        >>> # Next we need to set the hyperparameters for the training job. For example frequency of the time series used, number of data points the model will look at in the past, number of predicted data points. The other hyperparameters concern the model to train (number of layers, number of cells per layer, likelihood function) and the training options (number of epochs, batch size, learning rate...). We use default parameters for every optional parameter in this case (you can always use [Sagemaker Automated Model Tuning](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/) to tune them).

        >>> hyperparameters = {
        >>>     "time_freq": freq,
        >>>     "epochs": "400",
        >>>     "early_stopping_patience": "40",
        >>>     "mini_batch_size": "64",
        >>>     "learning_rate": "5E-4",
        >>>     "context_length": str(context_length),
        >>>     "prediction_length": str(prediction_length),
        >>> }


        >>> estimator.set_hyperparameters(**hyperparameters)


        >>> # We are ready to launch the training job. SageMaker will start an EC2 instance, download the data from S3, start training the model and save the trained model.
        >>> # 
        >>> # If you provide the `test` data channel as we do in this example, DeepAR will also calculate accuracy metrics for the trained model on this test. This is done by predicting the last `prediction_length` points of each time-series in the test set and comparing this to the actual value of the time-series. 
        >>> # 
        >>> # **Note:** the next cell may take a few minutes to complete, depending on data size, model complexity, training options.

        >>> get_ipython().run_cell_magic('time', '', 'data_channels = {"train": "{}/train/".format(s3_data_path), "test": "{}/test/".format(s3_data_path)}\n\nestimator.fit(inputs=data_channels, wait=True)\n')

        >>> %%time
        >>> data_channels = {"train": "{}/train/".format(s3_data_path), "test": "{}/test/".format(s3_data_path)}
        >>> 
        >>> estimator.fit(inputs=data_channels, wait=True)

            2024-10-31 23:01:59 Starting - Starting the training job...
            2024-10-31 23:02:14 Starting - Preparing the instances for training...
            2024-10-31 23:02:49 Downloading - Downloading the training image..................
            2024-10-31 23:06:02 Training - Training image download completed. Training in progress...Docker entrypoint called with argument(s): train
            Running default environment configuration script
            Running custom environment configuration script
            /opt/amazon/lib/python3.8/site-packages/mxnet/model.py:97: SyntaxWarning: "is" with a literal. Did you mean "=="?
              if num_device is 1 and 'dist' not in kvstore:
            [10/31/2024 23:06:17 INFO 140186099169088] Reading default configuration from /opt/amazon/lib/python3.8/site-packages/algorithm/resources/default-input.json: {'_kvstore': 'auto', '_num_gpus': 'auto', '_num_kv_servers': 'auto', '_tuning_objective_metric': '', 'cardinality': 'auto', 'dropout_rate': '0.10', 'early_stopping_patience': '', 'embedding_dimension': '10', 'learning_rate': '0.001', 'likelihood': 'student-t', 'mini_batch_size': '128', 'num_cells': '40', 'num_dynamic_feat': 'auto', 'num_eval_samples': '100', 'num_layers': '2', 'test_quantiles': '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]'}
            [10/31/2024 23:06:17 INFO 140186099169088] Merging with provided configuration from /opt/ml/input/config/hyperparameters.json: {'context_length': '84', 'early_stopping_patience': '40', 'epochs': '400', 'learning_rate': '5E-4', 'mini_batch_size': '64', 'prediction_length': '84', 'time_freq': '2H'}
            [10/31/2024 23:06:17 INFO 140186099169088] Final configuration: {'_kvstore': 'auto', '_num_gpus': 'auto', '_num_kv_servers': 'auto', '_tuning_objective_metric': '', 'cardinality': 'auto', 'dropout_rate': '0.10', 'early_stopping_patience': '40', 'embedding_dimension': '10', 'learning_rate': '5E-4', 'likelihood': 'student-t', 'mini_batch_size': '64', 'num_cells': '40', 'num_dynamic_feat': 'auto', 'num_eval_samples': '100', 'num_layers': '2', 'test_quantiles': '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]', 'context_length': '84', 'epochs': '400', 'prediction_length': '84', 'time_freq': '2H'}
            Process 7 is a worker.
            . . .
            . . .
            . . .
            #metrics {"StartTime": 1730416744.9600978, "EndTime": 1730416744.9830174, "Dimensions": {"Algorithm": "AWS/DeepAR", "Host": "algo-1", "Operation": "training"}, "Metrics": {"state.deserialize.time": {"sum": 22.62592315673828, "count": 1, "min": 22.62592315673828, "max": 22.62592315673828}}}
            [10/31/2024 23:19:04 INFO 140186099169088] stopping training now
            [10/31/2024 23:19:04 INFO 140186099169088] #progress_metric: host=algo-1, completed 100 % of epochs
            [10/31/2024 23:19:04 INFO 140186099169088] Final loss: 3.611852537501942 (occurred at epoch 111)
            [10/31/2024 23:19:04 INFO 140186099169088] #quality_metric: host=algo-1, train final_loss <loss>=3.611852537501942
            [10/31/2024 23:19:04 INFO 140186099169088] Worker algo-1 finished training.
            [10/31/2024 23:19:04 WARNING 140186099169088] wait_for_all_workers will not sync workers since the kv store is not running distributed
            [10/31/2024 23:19:04 INFO 140186099169088] All workers finished. Serializing model for prediction.
            #metrics {"StartTime": 1730416744.983081, "EndTime": 1730416745.547561, "Dimensions": {"Algorithm": "AWS/DeepAR", "Host": "algo-1", "Operation": "training"}, "Metrics": {"get_graph.time": {"sum": 564.0559196472168, "count": 1, "min": 564.0559196472168, "max": 564.0559196472168}}}
            [10/31/2024 23:19:05 INFO 140186099169088] Number of GPUs being used: 0
            #metrics {"StartTime": 1730416745.5476255, "EndTime": 1730416745.7459705, "Dimensions": {"Algorithm": "AWS/DeepAR", "Host": "algo-1", "Operation": "training"}, "Metrics": {"finalize.time": {"sum": 762.5062465667725, "count": 1, "min": 762.5062465667725, "max": 762.5062465667725}}}
            [10/31/2024 23:19:05 INFO 140186099169088] Serializing to /opt/ml/model/model_algo-1
            [10/31/2024 23:19:05 INFO 140186099169088] Saved checkpoint to "/opt/ml/model/model_algo-1-0000.params"
            #metrics {"StartTime": 1730416745.7460227, "EndTime": 1730416745.7777426, "Dimensions": {"Algorithm": "AWS/DeepAR", "Host": "algo-1", "Operation": "training"}, "Metrics": {"model.serialize.time": {"sum": 31.673669815063477, "count": 1, "min": 31.673669815063477, "max": 31.673669815063477}}}
            [10/31/2024 23:19:05 INFO 140186099169088] Successfully serialized the model for prediction.
            [10/31/2024 23:19:05 INFO 140186099169088] #memory_usage::<batchbuffer> = 19.89501953125 mb
            [10/31/2024 23:19:05 INFO 140186099169088] Evaluating model accuracy on testset using 100 samples
            #metrics {"StartTime": 1730416745.777792, "EndTime": 1730416745.7812219, "Dimensions": {"Algorithm": "AWS/DeepAR", "Host": "algo-1", "Operation": "training"}, "Metrics": {"model.bind.time": {"sum": 0.0324249267578125, "count": 1, "min": 0.0324249267578125, "max": 0.0324249267578125}}}
            [10/31/2024 23:19:16 INFO 140186099169088] Number of test batches scored: 10
            [10/31/2024 23:19:27 INFO 140186099169088] Number of test batches scored: 20
            #metrics {"StartTime": 1730416745.7812662, "EndTime": 1730416771.5043879, "Dimensions": {"Algorithm": "AWS/DeepAR", "Host": "algo-1", "Operation": "training"}, "Metrics": {"model.score.time": {"sum": 25723.198175430298, "count": 1, "min": 25723.198175430298, "max": 25723.198175430298}}}
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, RMSE): 413.51135399130993
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, mean_absolute_QuantileLoss): 7124648.37836553
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, mean_wQuantileLoss): 0.1124704480697371
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.1]): 0.11228146486661573
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.2]): 0.1264752049660226
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.3]): 0.1322487106040305
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.4]): 0.13267061690563117
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.5]): 0.1285121270097538
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.6]): 0.12033310512302084
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.7]): 0.10748674721108034
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.8]): 0.089292398505035
            [10/31/2024 23:19:31 INFO 140186099169088] #test_score (algo-1, wQuantileLoss[0.9]): 0.06293365743644401
            [10/31/2024 23:19:31 INFO 140186099169088] #quality_metric: host=algo-1, test RMSE <loss>=413.51135399130993
            [10/31/2024 23:19:31 INFO 140186099169088] #quality_metric: host=algo-1, test mean_wQuantileLoss <loss>=0.1124704480697371
            #metrics {"StartTime": 1730416771.5044594, "EndTime": 1730416771.5674973, "Dimensions": {"Algorithm": "AWS/DeepAR", "Host": "algo-1", "Operation": "training"}, "Metrics": {"setuptime": {"sum": 3.7603378295898438, "count": 1, "min": 3.7603378295898438, "max": 3.7603378295898438}, "totaltime": {"sum": 793738.3847236633, "count": 1, "min": 793738.3847236633, "max": 793738.3847236633}}}
            
            
            2024-10-31 23:19:40 Uploading - Uploading generated training model
            2024-10-31 23:19:48 Completed - Training job completed
            Training seconds: 1033
            Billable seconds: 1033
            CPU times: user 2.65 s, sys: 76.3 ms, total: 2.73 s
            Wall time: 18min 28s


        >>> # Since you pass a test set in this example, accuracy metrics for the forecast are computed and logged (see bottom of the log).
        >>> # You can find the definition of these metrics from [our documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html). You can use 
        >>> # these to optimize the parameters and tune your model or use SageMaker's [Automated Model Tuning service]
        >>> # (https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/) to tune the model for you.

        >>> # ### Create endpoint and predictor

        >>> # Now that we have a trained model, we can use it to perform predictions by deploying it to an endpoint.
        >>> # 
        >>> # **Note: Remember to delete the endpoint after running this experiment. A cell at the very bottom of this notebook will do that: 
        >>> # make sure you run it at the end.**

        >>> # To query the endpoint and perform predictions, we can define the following utility class: this allows making requests using 
        >>> # `pandas.Series` objects rather than raw JSON strings.

        >>> from sagemaker.serializers import IdentitySerializer


        >>> class DeepARPredictor(sagemaker.predictor.Predictor):
        >>>     def __init__(self, *args, **kwargs):
        >>>         super().__init__(
        >>>             *args,
        >>>             # serializer=JSONSerializer(),
        >>>             serializer=IdentitySerializer(content_type="application/json"),
        >>>             **kwargs,
        >>>         )

        >>>     def predict(
        >>>         self,
        >>>         ts,
        >>>         cat=None,
        >>>         dynamic_feat=None,
        >>>         num_samples=100,
        >>>         return_samples=False,
        >>>         quantiles=["0.1", "0.5", "0.9"],
        >>>     ):
        >>>         """Requests the prediction of for the time series listed in `ts`, each with the (optional)
        >>>         corresponding category listed in `cat`.

        >>>         ts -- `pandas.Series` object, the time series to predict
        >>>         cat -- integer, the group associated to the time series (default: None)
        >>>         num_samples -- integer, number of samples to compute at prediction time (default: 100)
        >>>         return_samples -- boolean indicating whether to include samples in the response (default: False)
        >>>         quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])

        >>>         Return value: list of `pandas.DataFrame` objects, each containing the predictions
        >>>         """
        >>>         prediction_time = ts.index[-1] + ts.index.freq
        >>>         quantiles = [str(q) for q in quantiles]
        >>>         req = self.__encode_request(ts, cat, dynamic_feat, num_samples, return_samples, quantiles)
        >>>         res = super(DeepARPredictor, self).predict(req)
        >>>         return self.__decode_response(res, ts.index.freq, prediction_time, return_samples)

        >>>     def __encode_request(self, ts, cat, dynamic_feat, num_samples, return_samples, quantiles):
        >>>         instance = series_to_dict(
        >>>             ts, cat if cat is not None else None, dynamic_feat if dynamic_feat else None
        >>>         )

        >>>         configuration = {
        >>>             "num_samples": num_samples,
        >>>             "output_types": ["quantiles", "samples"] if return_samples else ["quantiles"],
        >>>             "quantiles": quantiles,
        >>>         }

        >>>         http_request_data = {"instances": [instance], "configuration": configuration}

        >>>         return json.dumps(http_request_data).encode("utf-8")

        >>>     def __decode_response(self, response, freq, prediction_time, return_samples):
        >>>         # we only sent one time series so we only receive one in return
        >>>         # however, if possible one will pass multiple time series as predictions will then be faster
        >>>         predictions = json.loads(response.decode("utf-8"))["predictions"][0]
        >>>         prediction_length = len(next(iter(predictions["quantiles"].values())))
        >>>         prediction_index = pd.date_range(
        >>>             start=prediction_time, freq=freq, periods=prediction_length
        >>>         )
        >>>         if return_samples:
        >>>             dict_of_samples = {"sample_" + str(i): s for i, s in enumerate(predictions["samples"])}
        >>>         else:
        >>>             dict_of_samples = {}
        >>>         return pd.DataFrame(
        >>>             data={**predictions["quantiles"], **dict_of_samples}, index=prediction_index
        >>>         )

        >>>     def set_frequency(self, freq):
        >>>         self.freq = freq


        >>> def encode_target(ts):
        >>>     return [x if np.isfinite(x) else "NaN" for x in ts]


        >>> def series_to_dict(ts, cat=None, dynamic_feat=None):
        >>>     """Given a pandas.Series object, returns a dictionary encoding the time series.

        >>>     ts -- a pands.Series object with the target time series
        >>>     cat -- an integer indicating the time series category

        >>>     Return value: a dictionary
        >>>     """
        >>>     obj = {"start": str(ts.index[0]), "target": encode_target(ts)}
        >>>     if cat is not None:
        >>>         obj["cat"] = cat
        >>>     if dynamic_feat is not None:
        >>>         obj["dynamic_feat"] = dynamic_feat
        >>>     return obj


        >>> # Now we can deploy the model and create and endpoint that can be queried using our custom DeepARPredictor class.
            # specify predictor class with 'predictor_cls'
        >>> predictor = estimator.deploy(
        >>>     initial_instance_count=1, instance_type="ml.m5.large", predictor_cls=DeepARPredictor
        >>> )
            INFO:sagemaker:Creating model with name: deepar-electricity-demo-2024-11-01-01-57-09-029
            INFO:sagemaker:Creating endpoint-config with name deepar-electricity-demo-2024-11-01-01-57-09-029
            INFO:sagemaker:Creating endpoint with name deepar-electricity-demo-2024-11-01-01-57-09-029


        >>> # ### Make predictions and plot results

        >>> # Now we can use the `predictor` object to generate predictions.

        >>> predictor.predict(ts=timeseries[120], quantiles=[0.10, 0.5, 0.90]).head()


        >>> # Below we define a plotting function that queries the model and displays the forecast.

        >>> def plot(
        >>>     predictor,
        >>>     target_ts,
        >>>     cat=None,
        >>>     dynamic_feat=None,
        >>>     forecast_date=end_training,
        >>>     show_samples=False,
        >>>     plot_history=7 * 12,
        >>>     confidence=80,
        >>> ):
        >>>     freq = target_ts.index.freq
        >>>     print(
        >>>         "calling served model to generate predictions starting from {}".format(str(forecast_date))
        >>>     )
        >>>     assert confidence > 50 and confidence < 100
        >>>     low_quantile = 0.5 - confidence * 0.005
        >>>     up_quantile = confidence * 0.005 + 0.5

        >>>     # we first construct the argument to call our model
        >>>     args = {
        >>>         "ts": target_ts[:forecast_date],
        >>>         "return_samples": show_samples,
        >>>         "quantiles": [low_quantile, 0.5, up_quantile],
        >>>         "num_samples": 100,
        >>>     }

        >>>     if dynamic_feat is not None:
        >>>         args["dynamic_feat"] = dynamic_feat
        >>>         fig = plt.figure(figsize=(20, 6))
        >>>         ax = plt.subplot(2, 1, 1)
        >>>     else:
        >>>         fig = plt.figure(figsize=(20, 3))
        >>>         ax = plt.subplot(1, 1, 1)

        >>>     if cat is not None:
        >>>         args["cat"] = cat
        >>>         ax.text(0.9, 0.9, "cat = {}".format(cat), transform=ax.transAxes)

        >>>     # call the end point to get the prediction
        >>>     prediction = predictor.predict(**args)

        >>>     # plot the samples
        >>>     if show_samples:
        >>>         for key in prediction.keys():
        >>>             if "sample" in key:
        >>>                 prediction[key].plot(color="lightskyblue", alpha=0.2, label="_nolegend_")

        >>>     # plot the target
        >>>     target_section = target_ts[
        >>>         forecast_date - plot_history * freq : forecast_date + prediction_length * freq
        >>>     ]
        >>>     target_section.plot(color="black", label="target")

        >>>     # plot the confidence interval and the median predicted
        >>>     ax.fill_between(
        >>>         prediction[str(low_quantile)].index,
        >>>         prediction[str(low_quantile)].values,
        >>>         prediction[str(up_quantile)].values,
        >>>         color="b",
        >>>         alpha=0.3,
        >>>         label="{}% confidence interval".format(confidence),
        >>>     )
        >>>     prediction["0.5"].plot(color="b", label="P50")
        >>>     ax.legend(loc=2)

        >>>     # fix the scale as the samples may change it
        >>>     ax.set_ylim(target_section.min() * 0.5, target_section.max() * 1.5)

        >>>     if dynamic_feat is not None:
        >>>         for i, f in enumerate(dynamic_feat, start=1):
        >>>             ax = plt.subplot(len(dynamic_feat) * 2, 1, len(dynamic_feat) + i, sharex=ax)
        >>>             feat_ts = pd.Series(
        >>>                 index=pd.date_range(
        >>>                     start=target_ts.index[0], freq=target_ts.index.freq, periods=len(f)
        >>>                 ),
        >>>                 data=f,
        >>>             )
        >>>             feat_ts[
        >>>                 forecast_date - plot_history * freq : forecast_date + prediction_length * freq
        >>>             ].plot(ax=ax, color="g")


        >>> # We can interact with the function previously defined, to look at the forecast of any customer at any point in (future) time. 
        >>> # 
        >>> # For each request, the predictions are obtained by calling our served model on the fly.
        >>> # 
        >>> # Here we forecast the consumption of an office after week-end (note the lower week-end consumption). 
        >>> # You can select any time series and any forecast date, just click on `Run Interact` to generate the predictions from our served endpoint and see the plot.

        >>> style = {"description_width": "initial"}


        >>> @interact_manual(
        >>>     customer_id=IntSlider(min=0, max=369, value=91, style=style),
        >>>     forecast_day=IntSlider(min=0, max=100, value=51, style=style),
        >>>     confidence=IntSlider(min=60, max=95, value=80, step=5, style=style),
        >>>     history_weeks_plot=IntSlider(min=1, max=20, value=1, style=style),
        >>>     show_samples=Checkbox(value=False),
        >>>     continuous_update=False,
        >>> )
        >>> def plot_interact(customer_id, forecast_day, confidence, history_weeks_plot, show_samples):
        >>>     plot(
        >>>         predictor,
        >>>         target_ts=timeseries[customer_id],
        >>>         forecast_date=end_training + datetime.timedelta(days=forecast_day),
        >>>         show_samples=show_samples,
        >>>         plot_history=history_weeks_plot * 12 * 7,
        >>>         confidence=confidence,
        >>>     )

        >>> # ### Delete endpoints

        >>> predictor.delete_model()
        >>> predictor.delete_endpoint()

        >>> # ## Additional features
        >>> # 
        >>> # We have seen how to prepare a dataset and run DeepAR for a simple example.
        >>> # 
        >>> # In addition DeepAR supports the following features:
        >>> # 
        >>> # * missing values: DeepAR can handle missing values in the time series during training as well as for inference.
        >>> # * Additional time features: DeepAR provides a set default time series features such as hour of day etc. However, you can provide additional feature time series via the `dynamic_feat` field. 
        >>> # * generalize frequencies: any integer multiple of the previously supported base frequencies (minutes `min`, hours `H`, days `D`, weeks `W`, month `M`) are now allowed; e.g., `15min`. We already demonstrated this above by using `2H` frequency.
        >>> # * categories: If your time series belong to different groups (e.g. types of product, regions, etc), this information can be encoded as one or more categorical features using the `cat` field.
        >>> # 
        >>> # We will now demonstrate the missing values and time features support. For this part we will reuse the electricity dataset but will do some artificial changes to demonstrate the new features: 
        >>> # * We will randomly mask parts of the time series to demonstrate the missing values support.
        >>> # * We will include a "special-day" that occurs at different days for different time series during this day we introduce a strong up-lift
        >>> # * We train the model on this dataset giving "special-day" as a custom time series feature





