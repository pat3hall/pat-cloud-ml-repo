
-----------------------------------------------------------------
From: O'Reilly Hands-ON Machine Learning with Scikit-Learn, Keras, and TensorFlow


4.1 Linear Regression (pages 132 - 137)

     Linear model:
      - makes a prediction by simple computing a weighted sum of the input features pus a constanct 
        called the 'bias term' (also called the intercept)

  Linear Regression (pages 132 - 134)
      Equation 4-1 Linear regression model prediction:

      pred-y = theta0 + theta1 * x1 + theta2 * x2 + ... + theta-n * xn
        pred-y: the predicted value
        n:      the number of features
        xi:     the ith feature value
        theta-j: the jth model parameter, including the bias term, theta0

    training linear regression model:
      - find the value of theta that minimizes the RMSE
      - but it is simplier to minimize the MSE (mean square error), but leads to the same result

    equation 4-3: MSE cost functon for a linear regression model

       MSE (X, h-theta) = (1/m) SUM (theta-T * xi - yi)2  where SUM is from i = 1 to m


  Normal Equation (pages 134 - 137)

    Normal Equation:
      - a closed-form solution to find theta that minimizes the MSE

    Equation 4-4 Normal Equation

     pred-theta = (XT @ X)**-1  @ XT @ y

        in this equation:
         pred-theta: the value of theta that minimizes the cost function
         y: the vector of the target values containing y1 to ym
         XT: matrix X transposed

      >>> theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    Example: Using Linear Regression classifer

     >>> from sklearn.linear_model import LinearRegression
     >>> 
     >>> lin_reg = LinearRegression()
     >>> lin_reg.fit(X, y)
     >>> lin_reg.intercept_, lin_reg.coef_

    LinearRegression Classifier:
      - separates the 'bias term' (intercept_) from the feature weight coefficient (coef_)
      - based on the scipy.linalg.lstsq() function 

  Computation Complexity (pages 137 - 138):
    - normal equation computes the inverse of XT * X, which is an (n + 1) x (n + 1) matrix (where n is the
      number of features.
      - computation complexity of inverting such a matrix  is typically about O(n**2.4) to O(n**3)
      - if you double the number of features, you multiple the computation time by roughly 2**2.4 = 5.3 to 2**3 = 8
      - the SVD approach used by the Scikit-Learn's LinearRegression class is about O(n**2


  Linear Learner hyperparameters
    "feature_dim":"14",
       - number of features in the input data (default: auto)
    "mini_batch_size":"100",
      - The number of observations per mini-batch for the data iterator. (default: 1000)
    "predictor_type":"regressor",
      - Specifies the type of target variable as a binary classification, multiclass classification, or regression.
      - Valid values: binary_classifier, multiclass_classifier, or regressor
    "epochs":"10",
      - The maximum number of passes over the training data. (default: 15)
    "num_models":"32",
      - recommend using 32 - cheap way to get better results (max value: 32)
      - the number of models to train in parallel. 
      - For the default, auto, the algorithm decides the number of parallel models to train. 
      - One model is trained according to the given training parameter (regularization, optimizer, loss), and the 
        rest by close parameters.
    "loss":"absolute_loss"
      - specify loss function
      - for "predictor_type":"regressor",
        -  options are auto, squared_loss, absolute_loss, eps_insensitive_squared_loss, eps_insensitive_absolute_loss, 
           quantile_loss, and huber_loss. 
        - The default value for auto is squared_loss.
      - for "predictor_type":"binary_classifier",
        - options are auto,logistic, and hinge_loss. 
        - The default value for auto is logistic. 
      - for "predictor_type":"multiclass_classifier",
        - options are auto and softmax_loss. 
        - The default value for auto is softmax_loss.

   S3DataSource
     S3DataDistributionType:
       - Valid Values: FullyReplicated | ShardedByS3Key
       - FullyReplicated : 
          - to replicate the entire dataset on each ML compute instance that is launched for model training,
       - ShardedByS3Key
          - to replicate a subset of data on each ML compute instance that is launched for model training,
  
      S3DataType  
        - Valid Values: ManifestFile | S3Prefix | AugmentedManifestFile
        - S3Prefix:
           - S3Uri identifies a key name prefix. 
           - Amazon SageMaker uses all objects that match the specified key name prefix for model training.
        - ManifestFile, 
           - S3Uri identifies an object that is a manifest file containing a list of object keys that you want 
              Amazon SageMaker to use for model training.
         - AugmentedManifestFile 
           - S3Uri identifies an object that is an augmented manifest file in JSON lines format. 
           - This file contains the data you want to use for model training.
           - AugmentedManifestFile can only be used if the Channel's input mode is Pipe.


  client.create_training_job(**request_body)
    - Starts a model training job. After training completes, Amazon SageMaker saves the resulting model 
      artifacts to an Amazon S3 location that you specify. 
    - request_body:  json syntax with:
       • AlgorithmSpecification - Identifies the training algorithm to use.
       • HyperParameters - Specify these algorithm-specific parameters to enable the estimation of model
         parameters during training. Hyperparameters can be tuned to optimize this learning process. 
       • InputDataConfig - Describes the training dataset and the Amazon S3, EFS, or FSx location 
          where it is stored.
       • OutputDataConfig - Identifies the Amazon S3 bucket where you want Amazon SageMaker to save
         the results of model training.
       • ResourceConfig - Identifies the resources, ML compute instances, and ML storage volumes to deploy
         for model training. In distributed training, you specify more than one instance.
       • EnableManagedSpotTraining - Optimize the cost of training machine learning models by up to
         80% by using Amazon EC2 Spot instances. For more information, see Managed Spot Training.
       • RoleARN - The Amazon Resource Number (ARN) that Amazon SageMaker assumes to perform tasks on
         your behalf during model training. 
       • StoppingCondition - To help cap training costs, use MaxRuntimeInSeconds to set a time limit for training. 

-----------------------------------------------------------------

Amazon SageMaker’s Built-in Algorithm Webinar Series: Linear Learner
  https://www.youtube.com/watch?v=ae08a6Bp5lM

Summary:
  The SageMaker built-in algorithm, Linear Learner, can train as a binary or multi-classification model as well as linear regression. 
  Join Chris Burns, AWS Partner Solution Architect, as he dives deep into the use cases for Linear Learner. 
  Learn more at - https://amzn.to/2OT1ppw.




  SageMaker built-in Linear Learner:
    - can train as a binary or mult classification model as well as linear regression
    - linear regression: Linear means we are going to make a prediction based on a linear function of the input features
    - choice of the model is make by the "predictor_type" parameter



  Linear Regression
    β = parital slope coefficients

     given a set of (simple) points" {(β_1, β_1),(β_2, β_2),...,(β_n, β_n)}

     Y = β0 + β1X + e
       β0 and β1 are two unknowns constants that represent the intercept and the slope. e = error (term)

     Y-hat = β_0-hat + β_1-hat * x
        Y-hat: represents a predicition of Y 


  Data Prep
    - Linear Learner accepts CSV or record-IO-wrappred protobuf
    - CSV - no headers and label in first column
    - recordIO-wrapped = best practice = Pipe Mode
    - check Correlation and Standard deviation

  Training
    Linear Learning = a distributed implementation of stochastic gradient descent
    - trains multipe models simultaneously L1 (Ordinary Least Squares) L2(Ridge Regression - add a constraint to the coefficient)
    - Various optimizers: Adam, Adagard, SGD
    - before training starts you should understand what your success criteria is
    - the linear learner support FIVE metrics
       - objective Loss - mean value of the loss function
          - Binary classificatiion = Logistic loss
          - linear regression = Squared Loss

       - Binary classification accuracy
       - Binary F beta (or F1)
       - Precision
       - Recall

  Confusion Matrix

                        |  Predicted Class     |     Predicted Class
                        |  Negative            |     Positive
                        |  (NOT FRAUD)         |     (FRAUD)
          --------------|----------------------|----------------------
          Actual Class  | True Negative (TN)   |  False Positive (FP)       
                        |                      |                             
          Negative      | NOT FRAUD was        | FRAUD was incorrectly      
          (NOT FRAUD)   | correctly predicted  | predicted as NOT FRAUD    ^
                        | as NOT FRAUD         |                           |
          --------------|----------------------|----------------------     |
          Actual Class  | False Negative (FN)  |  True Positive (TP)       |
                        |                      |                           |
          Postive       | FRAUD was incorrectly| FRAUD was correctly       |
          (FRAUD)       | predicted as         | predicted as FRAUD       Precision
                        | NOT FRAUD            |                             
          --------------|----------------------|----------------------      
                                                  <-------- Recall

   Accuracy = Number of 


       Metric for Classification Problems

           Accuracy: (TP + TN)  / (TP + FP + TN + FN)
              - number of correct True Positives and True Negatives from all observations
              - percentage of predictions that were correct:
              - less effective with a lot of true negatives
                 - example: predicting fraud with little to no fraud data

           Precision: (TP)  / (TP + FP)
              - of all our predicted Positives observations, what precent are actually positive
              - accuracy of positive predictions
              - percentage of positive predictions that were correct:
              - Use when the cost of false positives is high
                 - example: an email is flagged and deleted as spam when it really isn't

           Recall: (TP)  / (TP + FN)
              - of all actual positives, what percent did we get correc5
              - also called sensitivity or true positive rate (TPR)
              - percentage of actual positive predictions that were correctly identified:
              - Use when the cost of false negatives is high
                 - example: someone has cancer, but screening does not find it

           F1 Score: (TP)  / [TP + ((FN + FP) / 2)]
             - combined precision and recall score
             - harmonic mean of the precision and recall 
             - regular mean treats all values equally, the harmonic mean give more weight to low values
             - classifiers will only get high F1 Score if both recall and precision values are high
             - train to F1 when you need a balance between False Negatives and False Positives

           Equation 3-3: F1 score:

           F1 = 2 / [ (1/precision) + (1/recall)]  =  2 x [( precision x recall) / (precision + recall)] 

              = (TP)  / [TP + ((FN + FP) / 2)]



  Code:

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
inport boto3
import re
from sagemaker import get_execution_role
import os
import io
import time
import json
import sagemaker.amazon.common as smac

role = get_execution role()
bucket = 'pat-demo-bkt1'
prefix = "housing"


s3 = boto3.resource('s3')
KEY = prefix + '/housing_boston.csv'
# download file to local drive
s3.Bucket(bucket).download_file(KEY, 'housing_boston.csv')

# read CSV to dataframe
data = pd.read_csv('housing_boston.csv')
# determine shape of the data
print(data.shape)
# or for pretty print view, use:o
display(data.head()))

# specify column names - canned datasets usually provide the column names and meanings
# target column generally is the 1st column
data.column = ["MDEV", "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
# print 1st 5 rows
print(data.head())

# for each column return the mean, std deviation, min, quartiles (25%, 50%, 75%), max
display(data.describe())

# display correlation info between the various feature (columns)
data.corr()


# generation heatmap (colormap) from the correlation data
#  format{("{:.2}") -> padding (spacing);  .background_gradient -> from mathplotlib
#  dark red for positive correlation, dark blue for high negative correlation
#  want to see either high positive or high negative correlation
data.corr().style.format("{:.2}").background_gradient(cmap=get_cmap('coolwarm'))


# split train data
rand_split = np.random.rand(len(data))

* split 80% training, 15% validation, & %5 testing
train_list = rand_split < 0.8
val_list   = (rand_split >= 0.8) & (rand_split < 0.95)
test_list  = (rand_split >= 0.95)

data_train = data[train_list]
data_val   = data[val_list]
data_test  = data[test_list]

# check train data
print(data_train)

# create train data and labels
# extract train labels (first column)
# goal -> buy any properties under $22K
# first column median value in $1K units
train_y = (data_train.iloc[:,0] > 22.0).as_matrix()
train_x = (data_train.iloc[:,1:].as_matrix()
# print out train_y label and train_x data shape
print('{},{}'.format(train_y.shape, train_x.shape))


# create valid data and labels
# extract validation labels (first column)
# goal -> buy any properties under $22K
# first column median value in $1K units
val_y = (data_val.iloc[:,0] > 22.0).as_matrix()
val_x = (data_val.iloc[:,1:].as_matrix()
# print out val_y label and val_x data shape
print('{},{}'.format(val_y.shape, val_x.shape))

# create test data and labels
# extract test labels (first column)
# goal -> buy any properties under $22K
# first column median value in $1K units
test_y = (data_test.iloc[:,0] > 22.0).as_matrix()
test_x = (data_test.iloc[:,1:].as_matrix()
# print out test_y label and test_x data shape
print('{},{}'.format(test_y.shape, test_x.shape))


# declare a file for training data
train_file = 'linear_train.data'
# convert to RecordIO-wrapped protobuf (using SageMaker Common library)
f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, train_x.astype('float32'), train_y.astype('float32'))
f.seek(0)
# upload training data to S3 to 'train' subfolder
S3.Bucket(bucket).Object(prefix + '/train/' + train_file).upload_fileobj(f)


# repeat for validation data
# declare a file for valid data
validation_file = 'linear_val.data'
# convert to RecordIO-wrapped protobuf (using SageMaker Common library)
f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, val_x.astype('float32'), val_y.astype('float32'))
f.seek(0)
# upload validation data to S3 to 'validation' subfolder
S3.Bucket(bucket).Object(prefix + '/validation/' + validation_file).upload_fileobj(f)


# declare a file for test data
test_file = 'linear_test.data'
# convert to RecordIO-wrapped protobuf (using SageMaker Common library)
f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, test_x.astype('float32'), test_y.astype('float32'))
f.seek(0)
# upload test data to S3 to 'test' subfolder
S3.Bucket(bucket).Object(prefix + '/test/' + test_file).upload_fileobj(f)


# get SageMaker LinearLearner container image
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri (boto3.Session().region_name, 'linear-learner')

linear_job = 'DeepDive-linear-' + time.strftime(%Y-%m-%d-%H-%M-%S", time.gmtime())

print ("Job name is: ", linear_job)

# specify linear parameters

linear_training_params = {
    "RoleArn": role,
    "TrainingJobName": linear_job,
    "AlgorithmSpecification": {
        "TrainingImage": container,
    "    TrainingInputMode": "File"
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.c4.2xlarge",
        "VolumeSizeInGB": 10
    },

    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "ShardedByS3Key"
                }
            },
            "CompressionType": "None"
            "RecordWrapperType": "None",
        },
        {
        "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None"
            "RecordWrapperType": "None",
        }
    ]
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/".format(bucket, prefix),
    },
    "HyperParameters": {
        "feature_dim":"14",
        "mini_batch_size":"100",
        "predictor_type":"regressor",
        "epochs":"10",
        "num_models":"32",
        "loss":"absolute_loss"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    },
}


# create and run training job
%%time

sm = boto3.client("sagemaker")
# create and start training job
sm.create_training_job(**linear_training_params)

status = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']
print(status)
# wait for model to complete (or fail)
sm.get_waiter('training_job_completed_or_stopped').wait((TrainingJobName=linear_job)
if status = 'Failed':
    messge = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']
    print(message)
    raise Exception('Training job failed')


# Prepare a model context for hosting to run inference
linear_hosting_container = {
    'Image': container,
    'ModelDataUri': sm.describe_training_job(TrainingJobName=linear_job)['ModelArtifacts']['S3ModelArtifacts']
}


#  Create a deployable model by identifying the location of model artifacts and the Docker image that
#    contains the inference code.
Create_model_response = sm.create_model(
    ModelName=linear_job,
    ExecutionRoleArn=role,
    PrimaryContainer=linear_hosting_container)

print(create_model_response['ModelArn'])


# Create an Amazon SageMaker endpoint configuration by specifying the ML compute instances that
#   you want to deploy your model to.
linear_endpoint_config = 'DeepDive-linear-endpoint-config-' + time.strftime(%Y-%m-%d-%H-%M-%S", time.gmtime())
print(linear_endpoint_config)


create_endpoint_config_response = sm.create_endpoint_config(
   EndpointConfigName = linear_endpoint_config,
   ProductionVariants=[{
       'InstanceType':'ml.m4.xlarge',
       'InitialInstanceCount':1,
       'ModelName': linear_job,
       'VariantName':'AllTraffic'}])
print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


#  Create an Amazon SageMaker endpoint. - deploy endpoint to SageMaker hosting
%%time

linear_endpoint = 'DeepDive-linear-endpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(linear_endpoint)

create_endpoint_response = sm.create_endpoint(
    EndpointName=linear_endpoint,
    EndpointConfigName=linear_endpoint_config)
print(create_endpoint_response['EndpointArn'])

resp = sm.describe_endpoint(EndpointName=linear_endpoint)
status = resp['EndpointStatus']
print("Status: " + status)

sm.get_waiter('endpoint_in_service').wait(EndpointName=linear_endpoint)
resp = sm.describe_endpoint(EndpointName=linear_endpoint)
status = resp['EndpointStatus']
print("Arn: " + resp['EndpointArn'])
print("Status: " + status)


# create helper function to convert an array to CSV
def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    f = open("test_observations.csv", 'w')
    f.write(csv.getvalue().decode())
    return csv.getvalue().decode().rstrip()

# Lets send our test file (as a .CSV file) to the endpoint to get prediction

# use endpoint=
runtime = boto3.client('runtime.sagemaker')

payload = np2csv(test_x)

response = runtime.invoke_endpoint(EndpointName=linear_endpoint, ContentType='text/csv', Body=payload)

result = json.loads(response['Body'].read().decode())
test_pred = np.array([r{'score'] for r in result['prediction']])


# Classification threshold of 0.5

test_pred_class = (test_pred > 0.5) + 0
test_pred_baseline = np.repeat(np.median(train_y), len(test_y))

prediction_accuracy = np.mean((test_y == test_pred_class))*100
baseline_accuracy =   np.mean((test_y == test_pred_baseline))*100

print('Prediction Accuracy"', round(prediction_accuracy,1), "%")
print('Baseline Accuracy"', round(baseline_accuracy,1), "%")
