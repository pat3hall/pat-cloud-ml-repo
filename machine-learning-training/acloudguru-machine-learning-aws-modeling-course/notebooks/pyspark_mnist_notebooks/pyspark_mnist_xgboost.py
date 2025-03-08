import os

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import sagemaker
from sagemaker import get_execution_role
import sagemaker_pyspark

role = get_execution_role()

# Configure Spark to use the SageMaker Spark dependency jars
jars = sagemaker_pyspark.classpath_jars()

classpath = ":".join(sagemaker_pyspark.classpath_jars())

# See the SageMaker Spark Github repo under sagemaker-pyspark-sdk
# to learn how to connect to a remote EMR cluster running Spark from a Notebook Instance.
spark = (
    SparkSession.builder.config("spark.driver.extraClassPath", classpath)
    .master("local[*]")
    .getOrCreate()
)

import boto3

cn_regions = ["cn-north-1", "cn-northwest-1"]
region = boto3.Session().region_name
endpoint_domain = "com.cn" if region in cn_regions else "com"
spark._jsc.hadoopConfiguration().set(
    "fs.s3a.endpoint", "s3.{}.amazonaws.{}".format(region, endpoint_domain)
)

trainingData = (
    spark.read.format("libsvm")
    .option("numFeatures", "784")
    .option("vectorType", "dense")
    .load("s3a://sagemaker-sample-data-{}/spark/mnist/train/".format(region))
)

testData = (
    spark.read.format("libsvm")
    .option("numFeatures", "784")
    .option("vectorType", "dense")
    .load("s3a://sagemaker-sample-data-{}/spark/mnist/test/".format(region))
)

trainingData.show()

import random
from sagemaker_pyspark import IAMRole, S3DataPath
from sagemaker_pyspark.algorithms import XGBoostSageMakerEstimator

xgboost_estimator = XGBoostSageMakerEstimator(
    sagemakerRole=IAMRole(role),
    trainingInstanceType="ml.m4.xlarge",
    trainingInstanceCount=1,
    endpointInstanceType="ml.m4.xlarge",
    endpointInitialInstanceCount=1,
)

xgboost_estimator.setEta(0.2)
xgboost_estimator.setGamma(4)
xgboost_estimator.setMinChildWeight(6)
xgboost_estimator.setSilent(0)
xgboost_estimator.setObjective("multi:softmax")
xgboost_estimator.setNumClasses(10)
xgboost_estimator.setNumRound(10)

# train
model = xgboost_estimator.fit(trainingData)

transformedData = model.transform(testData)

transformedData.show()

from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import numpy as np

# helper function to display a digit
def show_digit(img, caption="", xlabel="", subplot=None):
    if subplot == None:
        _, (subplot) = plt.subplots(1, 1)
    imgr = img.reshape((28, 28))
    subplot.axes.get_xaxis().set_ticks([])
    subplot.axes.get_yaxis().set_ticks([])
    plt.title(caption)
    plt.xlabel(xlabel)
    subplot.imshow(imgr, cmap="gray")


images = np.array(transformedData.select("features").cache().take(250))
clusters = transformedData.select("prediction").cache().take(250)

for cluster in range(10):
    print("\n\n\nCluster {}:".format(int(cluster)))
    digits = [img for l, img in zip(clusters, images) if int(l.prediction) == cluster]
    height = ((len(digits) - 1) // 5) + 1
    width = 5
    plt.rcParams["figure.figsize"] = (width, height)
    _, subplots = plt.subplots(height, width)
    subplots = np.ndarray.flatten(subplots)
    for subplot, image in zip(subplots, digits):
        show_digit(image, subplot=subplot)
    for subplot in subplots[len(digits) :]:
        subplot.axis("off")

    plt.show()

# Delete the endpoint

from sagemaker_pyspark import SageMakerResourceCleanup

resource_cleanup = SageMakerResourceCleanup(model.sagemakerClient)
resource_cleanup.deleteResources(model.getCreatedResources())
