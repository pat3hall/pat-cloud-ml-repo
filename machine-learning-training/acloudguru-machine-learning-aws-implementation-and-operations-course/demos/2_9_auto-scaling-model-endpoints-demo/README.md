------------------------------------------------------
2.9 DEMO: Automatically Scaling Model Endpoints

  automatically scaling model endpoints.
    - auto scaling is used by SageMaker to deliver high availability and fault tolerance for model endpoints.
  demo:
    - deploy a model using a Jupyter Notebook in SageMaker.
    - deploy a pre-trained model using some sample data.
    - configure auto scaling, and we'll stress test our model,
    - check the status of the endpoint to observe the instance count changing.


    -> SageMaker AI -> Applications and IDEs -> Studio -> [requires a domain to already exist] -> open studio

        -> <left> -> JupyterLab -> <upper right> +Create JupyterLab Space ->
         Name: MyJupyterLab, Sharing: Private -> create space
         # defaults: instance: ml.t3.medium, Image: SageMaker Distribution 1.9 (used 2.2.1)
         -> Run space # creates a Jupyter Notebook
         -> Open JupterLab

         # download git repo:
         git -> git clone repo -> Git repo URL: https://github.com/pluralsight-cloud/mls-c01-aws-certified-machine-learning-implementation-operations.git ,  unselect "Open Readm files" -> clone

          <under files> -> click on  auto-scaling-model-endpoints-demo/sagemaker_endpoint_demo.ipynb


    -> demo related files are provided under demos/2_9_auto-scaling-model-endpoints-demo/ :
          -> jupyter notebook:
             sagemaker_endpoint_demo.ipynb
          -> extracted python from jupyter notebook:
             sagemaker_endpoint_demo.py
          -> html view from completed jupyter notebook:
             sagemaker_endpoint_demo.html


     SageMaker / Client / create_endpoint_config
     https://boto3.amazonaws.com/v1/documentation/api/1.35.9/reference/services/sagemaker/client/create_endpoint_config.html

     ProductionVariant dict fields:
       VariantName (string) – [REQUIRED]
         - The name of the production variant.
       ModelName (string) –
         - The name of the model that you want to host. This is the name that you specified when creating the model.
       InitialInstanceCount (integer) –
         - Number of instances to launch initially.
       InstanceType (string) –
         - The ML compute instance type.
       InitialVariantWeight (float) –
         - Determines initial traffic distribution among all of the models that you specify in the endpoint configuration.
         - The traffic to a production variant is determined by the ratio of the VariantWeight to the sum of all VariantWeight
           values across all ProductionVariants.
         - If unspecified, it defaults to 1.0.

         - if two models, both have weight of 1.0, then 50% of the traffic will be sent to each of the 2 models


  Set Auto Scaling Policies for Multi-Model Endpoint Deployments
    https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints-autoscaling.html

    Define a scaling policy
      - To specify the metrics and target values for a scaling policy, you can configure a target-tracking scaling policy.
      - You can use either a predefined metric or a custom metric.
      - Scaling policy configuration is represented by a JSON block. You save your scaling policy configuration as a JSON block in a text file.

    Use a predefined metric
      - To quickly define a target-tracking scaling policy for a variant, use the 'SageMakerVariantInvocationsPerInstance' predefined metric.
      - 'SageMakerVariantInvocationsPerInstance' is the average number of times per minute that each instance for a variant is invoked.

     - To use a predefined metric in a scaling policy, create a target tracking configuration for your policy.
     - In the target tracking configuration, include a PredefinedMetricSpecification for the predefined metric and a TargetValue for the target value of that metric.



   ------------------

    Code:  sagemaker_endpoint_demo

      >>> # # SageMaker Auto Scaling Model Endpoints Demo
      >>> #  ## Run this notebook using SageMaker Studio Jupyter Lab

      >>> # First install the aiobotocore package which provides an interface to the AWS services that we'll be using

      >>> %pip install --upgrade -q aiobotocore


      >>> # We also need to install s3fs which enables Python to work with S3

      >>> pip install s3fs


      >>> # Import the libararies we need to build and deploy our model, and configure some parameters, including locations
      >>> # for model artifacts in S3

      >>> import pandas as pd
      >>> import numpy as np
      >>> import boto3
      >>> import sagemaker
      >>> import time
      >>> import json
      >>> import io
      >>> from io import StringIO
      >>> import base64
      >>> import pprint
      >>> import re
      >>> import s3fs

      >>> from sagemaker.image_uris import retrieve

      >>> sess = sagemaker.Session()
      >>> write_bucket = sess.default_bucket()
      >>> write_prefix = "fraud-detect-demo"

      >>> region = sess.boto_region_name
      >>> s3_client = boto3.client("s3", region_name=region)
      >>> sm_client = boto3.client("sagemaker", region_name=region)
      >>> sm_runtime_client = boto3.client("sagemaker-runtime")
      >>> sm_autoscaling_client = boto3.client("application-autoscaling")

      >>> sagemaker_role = sagemaker.get_execution_role()


      >>> # S3 locations used for parameterizing the notebook run
      >>> read_bucket = "sagemaker-sample-files"
      >>> read_prefix = "datasets/tabular/synthetic_automobile_claims"
      >>> model_prefix = "models/xgb-fraud"

      >>> data_capture_key = f"{write_prefix}/data-capture"

      >>> # S3 location of trained model artifact
      >>> model_uri = f"s3://{read_bucket}/{model_prefix}/fraud-det-xgb-model.tar.gz"

      >>> # S3 path where data captured at endpoint will be stored
      >>> data_capture_uri = f"s3://{write_bucket}/{data_capture_key}"

      >>> # S3 location of test data
      >>> test_data_uri = f"s3://{read_bucket}/{read_prefix}/test.csv"


      >>> # We're using the SageMaker managed XGBoost image

      >>> # Retrieve the SageMaker managed XGBoost image
      >>> training_image = retrieve(framework="xgboost", region=region, version="1.3-1")

      >>> # Specify a unique model name that does not exist
      >>> model_name = "fraud-detect-xgb"
      >>> primary_container = {
      >>>                      "Image": training_image,
      >>>                      "ModelDataUrl": model_uri
      >>>                     }

      >>> model_matches = sm_client.list_models(NameContains=model_name)["Models"]
      >>> if not model_matches:
      >>>     model = sm_client.create_model(ModelName=model_name,
      >>>                                    PrimaryContainer=primary_container,
      >>>                                    ExecutionRoleArn=sagemaker_role)
      >>> else:
      >>>     print(f"Model with name {model_name} already exists! Change model name to create new")


      >>> # Here's our endpoint configuration, including instance count, and instance type

      >>> # Endpoint Config name
      >>> endpoint_config_name = f"{model_name}-endpoint-config"

      >>> # Endpoint config parameters
      >>> production_variant_dict = {
      >>>                            "VariantName": "Alltraffic",
      >>>                            "ModelName": model_name,
      >>>                            "InitialInstanceCount": 1,
      >>>                            "InstanceType": "ml.m5.large",
      >>>                            "InitialVariantWeight": 1
      >>>                           }

      >>> # Data capture config parameters
      >>> data_capture_config_dict = {
      >>>                             "EnableCapture": True,
      >>>                             "InitialSamplingPercentage": 100,
      >>>                             "DestinationS3Uri": data_capture_uri,
      >>>                             "CaptureOptions": [{"CaptureMode" : "Input"}, {"CaptureMode" : "Output"}]
      >>>                            }


      >>> # Create endpoint config if one with the same name does not exist
      >>> endpoint_config_matches = sm_client.list_endpoint_configs(NameContains=endpoint_config_name)["EndpointConfigs"]
      >>> if not endpoint_config_matches:
      >>>     endpoint_config_response = sm_client.create_endpoint_config(
      >>>                                                                 EndpointConfigName=endpoint_config_name,
      >>>                                                                 ProductionVariants=[production_variant_dict],
      >>>                                                                 DataCaptureConfig=data_capture_config_dict
      >>>                                                                )
      >>> else:
      >>> 		print(f"Endpoint config with name {endpoint_config_name} already exists! Change endpoint config name to create new")


      >>> # Next, we deploy the model by creating the endpoint using the endpoint configuration that we created, it takes
      >>> # about 6 minutes to deploy.

      >>> endpoint_name = f"{model_name}-endpoint"

      >>> endpoint_matches = sm_client.list_endpoints(NameContains=endpoint_name)["Endpoints"]
      >>> if not endpoint_matches:
      >>>     endpoint_response = sm_client.create_endpoint(
      >>>                                                   EndpointName=endpoint_name,
      >>>                                                   EndpointConfigName=endpoint_config_name
      >>>                                                  )
      >>> else:
      >>>     print(f"Endpoint with name {endpoint_name} already exists! Change endpoint name to create new")

      >>> resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
      >>> status = resp["EndpointStatus"]
      >>> while status == "Creating":
      >>>     print(f"Endpoint Status: {status}...")
      >>>     time.sleep(60)
      >>>     resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
      >>>     status = resp["EndpointStatus"]
      >>> print(f"Endpoint Status: {status}")


      >>> # Invoke the endpoint by running some predictions using some sample data that is formatted using serialization
      >>> # and deserialization.

      >>> # Fetch test data to run predictions with the endpoint
      >>> test_df = pd.read_csv(test_data_uri)

      >>> # For content type text/csv, payload should be a string with commas separating the values for each feature
      >>> # This is the inference request serialization step
      >>> # CSV serialization
      >>> csv_file = io.StringIO()
      >>> test_sample = test_df.drop(["fraud"], axis=1).iloc[:5]
      >>> test_sample.to_csv(csv_file, sep=",", header=False, index=False)
      >>> payload = csv_file.getvalue()
      >>> response = sm_runtime_client.invoke_endpoint(
      >>>                                              EndpointName=endpoint_name,
      >>>                                              Body=payload,
      >>>                                              ContentType="text/csv",
      >>>                                              Accept="text/csv"
      >>>                                             )

      >>> # This is the inference response deserialization step
      >>> # This is a bytes object
      >>> result = response["Body"].read()
      >>> # Decoding bytes to a string
      >>> result = result.decode("utf-8")
      >>> # Converting to list of predictions
      >>> result = re.split(",|\n",result)

      >>> prediction_df = pd.DataFrame()
      >>> prediction_df["Prediction"] = result[:5]
      >>> prediction_df["Label"] = test_df["fraud"].iloc[:5].values
      >>> prediction_df


      >>> # Configure an auto scaling policy, minimum capacity is 1, maximum capacity is 2

      >>> resp = sm_client.describe_endpoint(EndpointName=endpoint_name)

      >>> # SageMaker expects resource id to be provided with the following structure
      >>> resource_id = f"endpoint/{endpoint_name}/variant/{resp['ProductionVariants'][0]['VariantName']}"

      >>> # Scaling configuration
      >>> scaling_config_response = sm_autoscaling_client.register_scalable_target(
      >>>                                                           ServiceNamespace="sagemaker",
      >>>                                                           ResourceId=resource_id,
      >>>                                                           ScalableDimension="sagemaker:variant:DesiredInstanceCount",
      >>>                                                           MinCapacity=1,
      >>>                                                           MaxCapacity=2
      >>>                                                         )


      >>> # Create the scaling policy. The scaling metric is SageMakerVariantInvocationsPerInstance (the average number
      >>> # of invocations per minute per model instance). When this number exceeds 5, auto scaling will be triggered.

      >>> # Create Scaling Policy
      >>> policy_name = f"scaling-policy-{endpoint_name}"
      >>> scaling_policy_response = sm_autoscaling_client.put_scaling_policy(
      >>>                                                 PolicyName=policy_name,
      >>>                                                 ServiceNamespace="sagemaker",
      >>>                                                 ResourceId=resource_id,
      >>>                                                 ScalableDimension="sagemaker:variant:DesiredInstanceCount",
      >>>                                                 PolicyType="TargetTrackingScaling",
      >>>                                                 TargetTrackingScalingPolicyConfiguration={
      >>>                                                     "TargetValue": 5.0, # Target for avg invocations per minutes
      >>>                                                     "PredefinedMetricSpecification": {
      >>>                                                         "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
      >>>                                                     },
      >>>                                                     "ScaleInCooldown": 600, # Duration in seconds until scale in
      >>>                                                     "ScaleOutCooldown": 60 # Duration in seconds between scale out
      >>>                                                 }
      >>>                                             )


      >>> # This code retrieves the scaling policy details

      >>> response = sm_autoscaling_client.describe_scaling_policies(ServiceNamespace="sagemaker")

      >>> pp = pprint.PrettyPrinter(indent=4, depth=4)
      >>> for i in response["ScalingPolicies"]:
      >>>     pp.pprint(i["PolicyName"])
      >>>     print("")
      >>>     if("TargetTrackingScalingPolicyConfiguration" in i):
      >>>         pp.pprint(i["TargetTrackingScalingPolicyConfiguration"])


      >>> # Stress test the endpoint to trigger auto scaling. This code runs for 250 seconds and repeatedly invokes the
      >>> # endpoint using random  samples from the test dataset.
      >>> #

      >>> request_duration = 250
      >>> end_time = time.time() + request_duration
      >>> print(f"Endpoint will be tested for {request_duration} seconds")
      >>> while time.time() < end_time:
      >>>     csv_file = io.StringIO()
      >>>     test_sample = test_df.drop(["fraud"], axis=1).iloc[[np.random.randint(0, test_df.shape[0])]]
      >>>     test_sample.to_csv(csv_file, sep=",", header=False, index=False)
      >>>     payload = csv_file.getvalue()
      >>>     response = sm_runtime_client.invoke_endpoint(
      >>>                                                  EndpointName=endpoint_name,
      >>>                                                  Body=payload,
      >>>                                                  ContentType="text/csv"
      >>>                                                 )


      >>> # Check the status of the endpoint

      >>> # Check the instance counts after the endpoint gets more load
      >>> response = sm_client.describe_endpoint(EndpointName=endpoint_name)
      >>> endpoint_status = response["EndpointStatus"]
      >>> request_duration = 250
      >>> end_time = time.time() + request_duration
      >>> print(f"Waiting for Instance count increase for a max of {request_duration} seconds. Please re run this cell in case the count does not change")
      >>> while time.time() < end_time:
      >>>     response = sm_client.describe_endpoint(EndpointName=endpoint_name)
      >>>     endpoint_status = response["EndpointStatus"]
      >>>     instance_count = response["ProductionVariants"][0]["CurrentInstanceCount"]
      >>>     print(f"Status: {endpoint_status}")
      >>>     print(f"Current Instance count: {instance_count}")
      >>>     if (endpoint_status=="InService") and (instance_count>1):
      >>>         break
      >>>     else:
      >>>         time.sleep(15)


      >>> # Delete model
      >>> sm_client.delete_model(ModelName=model_name)

      >>> # Delete endpoint configuration
      >>> sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

      >>> # Delete endpoint
      >>> sm_client.delete_endpoint(EndpointName=endpoint_name)


   ------------------


