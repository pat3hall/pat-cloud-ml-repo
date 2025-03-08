#!/usr/bin/env python
# coding: utf-8

# ![A Cloud Guru](acg_logo.png)
# <hr/>

# <center><h1>Training Reports Utilized in SageMaker Debugger to Improve Your Models</h1></center>

# # Introduction
# 
# In this lab, you will learn how to import the census dataset, which predicts whether an individual makes over 50K per year. The dataset is split into training and testing data and uploaded to an S3 bucket. Then, we fetch the xgboost algorithm and initialize the estimator object with the debugger hook, profiler config, and rules.

# # How to Use This Lab
# 
# Most of the code is provided for you in this lab as our solution to the tasks presented. Some of the cells are left empty with a #TODO header, and it's your turn to fill in the empty code. You can always use our lab guide if you are stuck.

# # 1) Install dependencies and import the required libraries

# In[1]:


# Install Sagemaker
get_ipython().system('pip install -U sagemaker smdebug numpy==1.26.4 shap')


# 1. The S3 bucket that you want to use for training data must be within the same region as the Notebook Instance.
# 2. The IAM role is used to provide training and hosting access to your data. See the documentation for how to create these. Note that if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with an appropriate full IAM role arn string(s).

# In[2]:


import sagemaker
import boto3
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.image_uris import retrieve
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sagemaker.debugger import Rule, DebuggerHookConfig, CollectionConfig, rule_configs, ProfilerConfig

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Define the S3 bucket and prefix to store data
output_bucket = sagemaker.Session().default_bucket()
output_prefix = 'sagemaker/xgboost-debugger'


# In[3]:


#TODO: Fetch the IAM role using the get_execution_role function and assign the value to a variable `role`
role = get_execution_role()


# # 2) Download the data and upload them to S3 bucket

# 1. The input data is downloaded from the `SHAP` library.
# 2. The data is split into training and testing data in the ratio of 80 - 20.
# 3. The data is saved under 'train.csv' and 'validation.csv'.

# In[4]:


import shap

X, y = shap.datasets.adult()


# In[5]:


#TODO: Use the `train_test_split` function and the split the data in a 80 - 20 ratio. 
#TODO: Assign the values to variables `X_train`, `X_test`, `y_train`, `y_test` 
# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# In[6]:


train_data = pd.concat(
    [pd.Series(y_train, index=X_train.index, name="Train data", dtype=int), X_train],
    axis=1,
)
validation_data = pd.concat(
    [pd.Series(y_test, index=X_test.index, name="Validation data", dtype=int), X_test],
    axis=1,
)


# In[7]:


X_train.index


# In[8]:


# Save to CSV
train_data.to_csv('train.csv', index=False, header=False)
validation_data.to_csv('validation.csv', index=False, header=False)


# Let's use the upload_file function and upload the .csv files to the S3 buckets.

# In[9]:


# Upload data to S3
s3 = boto3.client('s3')
s3.upload_file('train.csv', output_bucket, f'{output_prefix}/train/train.csv')


# In[10]:


#TODO: Using the strategy we followed to upload the training data as shown above, please upload the validation data to the output bucket.
s3.upload_file('validation.csv', output_bucket, f'{output_prefix}/validation/validation.csv')


# # 3) Set up training and validation data

# Create three separate variables that is dynamically constructed which will be used as one of the input parameters while generating training inputs.

# In[11]:


# creating the inputs for the fit() function with the training and validation location
s3_train_data = f"s3://{output_bucket}/{output_prefix}/train"
print(f"training files will be taken from: {s3_train_data}")
s3_validation_data = f"s3://{output_bucket}/{output_prefix}/validation"
print(f"validation files will be taken from: {s3_validation_data}")
output_location = f"s3://{output_bucket}/{output_prefix}/output"
print(f"training artifacts output location: {output_location}")


# Let's create the sagemaker.session.s3_input objects from our data channels. Note that we are using the content_type as text/csv. We use two channels here, one for training and the second for validation.

# In[12]:


# generating the session.s3_input() format for fit() accepted by the sdk
train_data = sagemaker.inputs.TrainingInput(
    s3_train_data,
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None,
)


# In[13]:


# Create the input parameter `validation_data`
validation_data = sagemaker.inputs.TrainingInput(
    s3_validation_data,
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None,
)


# # 4) Fetch the algorithm and initialize estimator

# Let's retrieve the image for the xgboost Algorithm according to the region.

# In[14]:


#TODO: Fetch the 1.2.1 version of xgboost algorithm according to the region and assign it to a variable container
container = retrieve('xgboost', boto3.Session().region_name, version='1.2-1')
print(container)
deploy_amt_model = True


# 1. The objective metric is set as binary logistic Since we are working on a binary classification problem. 
# 2. The variable `save_interval` is used in the estimator object to control the frequency of data collection.

# In[15]:


hyperparameters = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "objective": "binary:logistic",
    "num_round": "51",
}

save_interval = 5


# In[16]:


# Define Profiler configuration
profiler_config = ProfilerConfig(
    s3_output_path=f's3://{output_bucket}/{output_prefix}/profiler'  # Save Profiler output to S3
)


# 1. Then we create an estimator from the SageMaker Python SDK using the xgboost container image, and we set the training parameters. To turn on the sagemaker debugger, we need to add the debuggerhookconfig.The DebuggerHookConfig accepts one or more objects of type CollectionConfig, which defines the configuration around the tensor collection we intend to collect and save during model training. 
# 2. The next parameter is the debugger rules used by Amazon SageMaker Debugger to analyze metrics and tensors collected while training your models. The debugger's built-in rules monitor various common conditions critical for a training job's success. 

# In[17]:


from sagemaker.debugger import rule_configs, Rule, ProfilerRule, DebuggerHookConfig, CollectionConfig
from sagemaker.estimator import Estimator
from time import gmtime, strftime

job_name = "xgboost-iris-debugger-" + strftime("%Y%m%d-%H-%M-%S", gmtime())
print("Training job", job_name)

xgboost_estimator = Estimator(
    role=role,
    base_job_name=job_name,
    instance_count=1,
    instance_type="ml.m5.large",
    image_uri=container,
    hyperparameters=hyperparameters,
    max_run=1800,
    profiler_config=profiler_config,
    debugger_hook_config=DebuggerHookConfig(
        s3_output_path=output_location,  # Required
        collection_configs=[
            CollectionConfig(name="metrics", parameters={"save_interval": str(save_interval)}),
            CollectionConfig(
                name="feature_importance",
                parameters={"save_interval": str(save_interval)},
            ),            
        ],
    ),
    rules=[
        Rule.sagemaker(
            rule_configs.loss_not_decreasing(),
            rule_parameters={
                "collection_names": "metrics",
                "num_steps": str(save_interval * 2)                
            },
        ),
        Rule.sagemaker(rule_configs.create_xgboost_report()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport())
    ],
)


# In[18]:


#TODO: Invoke the fit function to initiate the debug enabled training process.
xgboost_estimator.fit({"train": train_data, "validation": validation_data})


# In[ ]:




