#!/usr/bin/env python
# coding: utf-8

# ![A Cloud Guru](acg_logo.png)
# <hr/>

# <center><h1>Linear Regression Performed Using Amazon SageMaker</h1></center>

# # Introduction
# 
# In this lab, you will learn how to import the iris dataset, split it into training and validation data, upload them to S3 bucket, fetch the linear learner algorithm, initialize the estimator object, set the hyperparameters and train the model.

# # How to Use This Lab
# 
# Most of the code is provided for you in this lab as our solution to the tasks presented. Some of the cells are left empty with a #TODO header and its your turn to fill in the empty code. You can always use our lab guide if you are stuck.

# # 1) Install dependencies and import the required libraries

# In[1]:


# Install Sagemaker
get_ipython().system('pip install boto3 sagemaker')


# 1. We will use the iris dataset as our input data. 
# 2. The S3 bucket that you want to use for training data must be within the same region as the Notebook Instance.
# 3. The IAM role is used to provide training and hosting access to your data. See the documentation for how to create these. Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with an appropriate full IAM role arn string(s).

# In[2]:


import sagemaker
import boto3
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Define the S3 bucket and prefix to store data
output_bucket = sagemaker.Session().default_bucket()
output_prefix = 'sagemaker/linear-learner'


# In[3]:


#TODO: Fetch the IAM role using the get_execution_role function and assign the value to a variable `role`
role = get_execution_role()


# # 2) Download the data and upload them to S3 bucket

# 1. load_iris function is used to download the input data
# 2. The data is split into training and validation data in the ratio of 80 - 20
# 3. The data is saved under 'train.csv' and 'validation.csv'

# In[4]:


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for easier manipulation
data = pd.DataFrame(X, columns=iris.feature_names)
data['target'] = y


# In[5]:


data.head()


# In[6]:


#TODO: Use the `train_test_split` function and split the data in a 80 - 20 ratio. 
#TODO: Assign the values to variables `train_data` and `validation_data` 
train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)


# In[7]:


# Save to CSV
train_data.to_csv('train.csv', index=False, header=False)
validation_data.to_csv('validation.csv', index=False, header=False)


# Let's use the upload_file function and upload the .csv files to the S3 buckets

# In[8]:


# Upload data to S3
s3 = boto3.client('s3')
s3.upload_file('train.csv', output_bucket, f'{output_prefix}/train/train.csv')


# In[9]:


#TODO: Using the strategy we followed to upload the training data, as shown above, please upload the validation data to the output bucket.
s3.upload_file('validation.csv', output_bucket, f'{output_prefix}/validation/validation.csv')


# # 3) Set up training and validation data

# Create three separate variables that are dynamically constructed, which will be used as one of the input parameters while generating training inputs.

# In[10]:


# creating the inputs for the fit() function with the training and validation location
s3_train_data = f"s3://{output_bucket}/{output_prefix}/train"
print(f"training files will be taken from: {s3_train_data}")
s3_validation_data = f"s3://{output_bucket}/{output_prefix}/validation"
print(f"validation files will be taken from: {s3_validation_data}")
output_location = f"s3://{output_bucket}/{output_prefix}/output"
print(f"training artifacts output location: {output_location}")


# Let's create the sagemaker.session.s3_input objects from our data channels. Note that we are using the content_type as text/csv. We use two channels here, one for training and the second for validation.

# In[11]:


# generating the session.s3_input() format for fit() accepted by the sdk
train_data = sagemaker.inputs.TrainingInput(
    s3_train_data,
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None,
)


# In[12]:


#TODO: Following the same strategy shown above, please set up a training input for validation data.
#TODO: Name it as `validation_data`
validation_data = sagemaker.inputs.TrainingInput(
    s3_validation_data,
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None,
)


# # 4) Fetch the algorithm and train the model

# Let's retrieve the image for the Linear Learner Algorithm according to the region.

# In[13]:


# Fetch the linear learner image according to the region
from sagemaker.image_uris import retrieve

container = retrieve("linear-learner", boto3.Session().region_name, version="1")
print(container)
deploy_amt_model = True


# Then, we create an estimator from the SageMaker Python SDK using the Linear Learner container image and set the training parameters.

# In[14]:


get_ipython().run_cell_magic('time', '', 'import boto3\nimport sagemaker\nfrom time import gmtime, strftime\n\nsess = sagemaker.Session()\n\njob_name = "linear-learner-iris-regression-" + strftime("%Y%m%d-%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\nlinear = sagemaker.estimator.Estimator(\n    container,\n    role,\n    instance_count=1,\n    instance_type="ml.m5.large",\n    output_path=output_location,\n    sagemaker_session=sagemaker_session,\n)\n')


# The hyperparameters are manually configured

# In[15]:


# TODO: Use the set_hyperparameters function and set the following hyperparameters on linear learner
# feature_dim=4, predictor_type='regressor', mini_batch_size=20
linear.set_hyperparameters(
    feature_dim=4,  # Adjust this to match your feature dimension
    predictor_type='regressor',  # Use 'classifier' for classification
    mini_batch_size=20
)


# 1. The following cell will train the algorithm. Training the algorithm involves a few steps. First, the instances that we requested while creating the Estimator classes are provisioned and set up with the appropriate libraries. Then, the data from our channels is downloaded into the instance. Once this is done, the training job begins. The provisioning and data downloading will take time, depending on the size of the data. Therefore, it might be a few minutes before we start getting data logs for our training jobs.
# 2. The log will print the objective metric details.
# 3. The training time takes between 4 and 6 minutes.

# In[16]:


get_ipython().run_cell_magic('time', '', 'linear.fit(inputs={"train": train_data, "validation": validation_data}, job_name=job_name)\n')


# In[ ]:




