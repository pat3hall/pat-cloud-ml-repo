#!/usr/bin/env python
# coding: utf-8

# # An Introduction to Linear Learner with MNIST
# 

# ---
# 
# This notebook's CI test result for us-west-2 is as follows. CI test results in other regions can be found at the end of the notebook. 
# 
# ![This us-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-2/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ---

# _**Making a Binary Prediction of Whether a Handwritten Digit is a 0**_
# 
# 1. [Introduction](#Introduction)
# 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
#   1. [Permissions and environment variables](#Permissions-and-environment-variables)
#   2. [Data ingestion](#Data-ingestion)
#   3. [Data inspection](#Data-inspection)
#   4. [Data conversion](#Data-conversion)
# 3. [Training the linear model](#Training-the-linear-model)
# 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
# 5. [Validate the model for use](#Validate-the-model-for-use)
# 

# ## Introduction
# 
# Welcome to our example introducing Amazon SageMaker's Linear Learner Algorithm!  Today, we're analyzing the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset which consists of images of handwritten digits, from zero to nine.  We'll use the individual pixel values from each 28 x 28 grayscale image to predict a yes or no label of whether the digit is a 0 or some other digit (1, 2, 3, ... 9).
# 
# The method that we'll use is a linear binary classifier.  Linear models are supervised learning algorithms used for solving either classification or regression problems.  As input, the model is given labeled examples ( **`x`**, `y`). **`x`** is a high dimensional vector and `y` is a numeric label.  Since we are doing binary classification, the algorithm expects the label to be either 0 or 1 (but Amazon SageMaker Linear Learner also supports regression on continuous values of `y`).  The algorithm learns a linear function, or linear threshold function for classification, mapping the vector **`x`** to an approximation of the label `y`.
# 
# Amazon SageMaker's Linear Learner algorithm extends upon typical linear models by training many models in parallel, in a computationally efficient manner.  Each model has a different set of hyperparameters, and then the algorithm finds the set that optimizes a specific criteria.  This can provide substantially more accurate models than typical linear algorithms at the same, or lower, cost.
# 
# To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.

# ## Prequisites and Preprocessing
# 
# The notebook works with *Data Science* kernel in SageMaker Studio.
# 
# ### Permissions and environment variables
# 
# _This notebook was created and tested on an ml.m4.xlarge notebook instance._
# 
# Let's start by specifying:
# 
# - The S3 bucket and prefix that you want to use for training and model data.  This should be within the same region as the Notebook Instance, training, and hosting.
# - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.  Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with a the appropriate full IAM role arn string(s).

# In[1]:


get_ipython().system(' pip install --upgrade sagemaker')


# In[27]:


import sagemaker

#bucket = sagemaker.Session().default_bucket()
bucket = "pat-demo-bkt-e2"
prefix = "sagemaker/DEMO-linear-mnist"

# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

sess = sagemaker.Session()
region = boto3.Session().region_name
print(f"region: {region}")

role = get_execution_role()
print (f"role: {role}")

# S3 bucket where the original mnist data is downloaded and stored.
# downloaded_data_bucket = f"sagemaker-example-files-prod-{region}"
# downloaded_data_prefix = "datasets/image/MNIST"


# ### Data ingestion
# 
# Next, we read the dataset from an online URL into memory, for preprocessing prior to training. This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.

# In[28]:


get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\nfobj = (\n    boto3.client("s3")\n    .get_object(\n        Bucket=f"sagemaker-example-files-prod-{boto3.session.Session().region_name}",\n        Key="datasets/image/MNIST/mnist.pkl.gz",\n    )["Body"]\n    .read()\n)\n\nwith open("mnist.pkl.gz", "wb") as f:\n    f.write(fobj)\n\n# Load the dataset\nwith gzip.open("mnist.pkl.gz", "rb") as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")\n')


# ### Data inspection
# 
# Once the dataset is imported, it's typical as part of the machine learning process to inspect the data, understand the distributions, and determine what type(s) of preprocessing might be needed. You can perform those tasks right here in the notebook. As an example, let's go ahead and look at one of the digits that is part of the dataset.

# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (2, 10)


def show_digit(img, caption="", subplot=None):
    if subplot == None:
        _, (subplot) = plt.subplots(1, 1)
    imgr = img.reshape((28, 28))
    subplot.axis("off")
    subplot.imshow(imgr, cmap="gray")
    plt.title(caption)


show_digit(train_set[0][30], "This is a {}".format(train_set[1][30]))


# ### Data conversion
# 
# Since algorithms have particular input and output requirements, converting the dataset is also part of the process that a data scientist goes through prior to initiating training. In this particular case, the Amazon SageMaker implementation of Linear Learner takes recordIO-wrapped protobuf, where the data we have today is a pickle-ized numpy array on disk.
# 
# Most of the conversion effort is handled by the Amazon SageMaker Python SDK, imported as `sagemaker` below.

# In[30]:


import io
import numpy as np
import sagemaker.amazon.common as smac

vectors = np.array([t.tolist() for t in train_set[0]]).astype("float32")
labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype("float32")

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)


# MINST dataset: 50000 images at 28 x 28 pixel per image
# labels: for image of '0', label[i] = 1, else 0

# In[31]:


vectors.size


# In[12]:


show_digit(train_set[0][21], "This is a {}".format(train_set[1][21]))


# ## Upload training data
# Now that we've created our recordIO-wrapped protobuf, we'll need to upload it to S3, so that Amazon SageMaker training can use it.

# In[32]:


import boto3
import os

key = "recordio-pb-data"
boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, "train", key)).upload_fileobj(buf)
s3_train_data = "s3://{}/{}/train/{}".format(bucket, prefix, key)
print("uploaded training data location: {}".format(s3_train_data))


# In[24]:





# Let's also setup an output S3 location for the model artifact that will be output as the result of training with the algorithm.

# In[33]:


output_location = "s3://{}/{}/output".format(bucket, prefix)
print("training artifacts will be uploaded to: {}".format(output_location))


# ## Training the linear model
# 
# Once we have the data preprocessed and available in the correct format for training, the next step is to actually train the model using the data. Since this data is relatively small, it isn't meant to show off the performance of the Linear Learner training algorithm, although we have tested it on multi-terabyte datasets.
# 
# Again, we'll use the Amazon SageMaker Python SDK to kick off training, and monitor status until it is completed.  In this example that takes between 7 and 11 minutes.  Despite the dataset being small, provisioning hardware and loading the algorithm container take time upfront.
# 
# First, let's specify our containers.  Since we want this notebook to run in all 4 of Amazon SageMaker's regions, we'll create a small lookup.  More details on algorithm containers can be found in [AWS documentation](https://docs-aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html).

# In[34]:


from sagemaker.image_uris import retrieve

container = retrieve("linear-learner", boto3.Session().region_name)


# In[35]:


session = boto3.Session()
s3 = session.resource('s3')

my_bucket = s3.Bucket(bucket)

for my_bucket_object in my_bucket.objects.all():
    print(my_bucket_object.key)


# Next we'll kick off the base estimator, making sure to pass in the necessary hyperparameters.  Notice:
# - `feature_dim` is set to 784, which is the number of pixels in each 28 x 28 image.
# - `predictor_type` is set to 'binary_classifier' since we are trying to predict whether the image is or is not a 0.
# - `mini_batch_size` is set to 200.  This value can be tuned for relatively minor improvements in fit and speed, but selecting a reasonable value relative to the dataset is appropriate in most cases.

# In[36]:


import boto3

sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(
    container,
    role,
    train_instance_count=1,
    train_instance_type="ml.c4.xlarge",
    output_path=output_location,
    sagemaker_session=sess,
)
linear.set_hyperparameters(feature_dim=784, predictor_type="binary_classifier", mini_batch_size=200)

linear.fit({"train": s3_train_data})


# ## Set up hosting for the model
# Now that we've trained our model, we can deploy it behind an Amazon SageMaker real-time hosted endpoint.  This will allow out to make predictions (or inference) from the model dyanamically.
# 
# _Note, Amazon SageMaker allows you the flexibility of importing models trained elsewhere, as well as the choice of not importing models if the target of model creation is AWS Lambda, AWS Greengrass, Amazon Redshift, Amazon Athena, or other deployment target._

# In[37]:


from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

linear_predictor = linear.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer(),
)


# ## Validate the model for use
# Finally, we can now validate the model for use.  We can pass HTTP POST requests to the endpoint to get back predictions.  To make this easier, we'll again use the Amazon SageMaker Python SDK and specify how to serialize requests and deserialize responses that are specific to the algorithm.

# Now let's try getting a prediction for a single record.

# In[38]:


result = linear_predictor.predict(train_set[0][30:31])
print(result)


# OK, a single prediction works.  We see that for one record our endpoint returned some JSON which contains `predictions`, including the `score` and `predicted_label`.  In this case, `score` will be a continuous value between [0, 1] representing the probability we think the digit is a 0 or not.  `predicted_label` will take a value of either `0` or `1` where (somewhat counterintuitively) `1` denotes that we predict the image is a 0, while `0` denotes that we are predicting the image is not of a 0.
# 
# Let's do a whole batch of images and evaluate our predictive accuracy.

# In[39]:


import numpy as np

predictions = []
for array in np.array_split(test_set[0], 100):
    result = linear_predictor.predict(array)
    predictions += [r["predicted_label"] for r in result["predictions"]]

predictions = np.array(predictions)


# In[40]:


import pandas as pd

pd.crosstab(
    np.where(test_set[1] == 0, 1, 0), predictions, rownames=["actuals"], colnames=["predictions"]
)


# As we can see from the confusion matrix above, we predict 931 images of 0 correctly, while we predict 44 images as 0s that aren't, and miss predicting 49 images of 0.

# ### (Optional) Delete the Endpoint
# 
# If you're ready to be done with this notebook, please run the delete_endpoint line in the cell below.  This will remove the hosted endpoint you created and avoid any charges from a stray instance being left on.

# In[41]:


sagemaker.Session().delete_endpoint(linear_predictor.endpoint)


# ## Notebook CI Test Results
# 
# This notebook was tested in multiple regions. The test results are as follows, except for us-west-2 which is shown at the top of the notebook.
# 
# ![This us-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This us-east-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-2/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This us-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This ca-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ca-central-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This sa-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/sa-east-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This eu-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This eu-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-2/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This eu-west-3 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-3/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This eu-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-central-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This eu-north-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-north-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This ap-southeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This ap-southeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-2/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This ap-northeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This ap-northeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-2/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
# ![This ap-south-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-south-1/aws_sagemaker_studio|sagemaker_algorithms|linear_learner_mnist|linear_learner_mnist.ipynb)
# 
