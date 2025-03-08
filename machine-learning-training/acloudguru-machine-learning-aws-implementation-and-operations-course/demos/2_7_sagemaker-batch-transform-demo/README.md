------------------------------------------------------
2.7 DEMO: Obtaining Inferences for an Entire Dataset Using SageMaker Batch Transform


  obtaining inferences for an entire data set using SageMaker Batch Transform
    - Batch Transform manages the processing of large data sets using one single operation,
    - a way to run inferences when you don't need a persistent endpoint.
    - split our data into training, validation, and batch inference.
    - train an XG Boost model
    - run a batch transform job


    -> SageMaker AI -> Applications and IDEs -> Studio -> [requires a domain to already exist] -> open studio
       # Note: This demo must be run on your own AWS account
       # needed steps to create domain
       -> Create a SageMaker domain -> select "Set up for single user (quick setup) -> setup


        -> <left> -> JupyterLab -> <upper right> +Create JupyterLab Space ->
         Name: MyJupyterLab, Sharing: Private -> create space
         # defaults: instance: ml.t3.medium, Image: SageMaker Distribution 1.9 (used 1.9.1)
         -> Run space # creates a Jupyter Notebook
         -> Open JupterLab

         # download git repo:
         git -> git clone repo -> Git repo URL: https://github.com/pluralsight-cloud/mls-c01-aws-certified-machine-learning-implementation-operations.git ,  unselect "Open Readm files" -> clone

          <under files> -> click on  sagemaker-batch-transform-demo/batch-transform-tumor-prediction.ipynb

       -> demo related files are provided under demos/2_7_sagemaker-batch-transform-demo/ :
          -> jupyter notebook:
          batch-transform-tumor-prediction.ipynb
          -> extracted python from jupyter notebook:
          batch-transform-tumor-prediction.py
          -> html view from completed jupyter notebook:
          batch-transform-tumor-prediction.html


    XGBoost hyperparameters used in demo:
      https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html

      objective="binary:logistic"
        - Specifies the learning task and the corresponding learning objective.
        - Examples:
            reg:logistic (logistic regression, output probability),
            multi:softmax (multiclass classification using the softmax objective),
            binary:logistic (logistic regression for binary classification, output probability)
            reg:squarederror (regression with squared loss)
        - For a full list of valid inputs, refer to XGBoost Learning Task Parameters
          https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst#learning-task-parameters
        - Optional;  Valid values: String; Default value: "reg:squarederror"

      max_depth=5
       - Maximum depth of a tree.
       - Increasing this value makes the model more complex and likely to be overfit.
       - 0 indicates no limit. A limit is required when grow_policy=depth-wise.
       - Optional;   Valid values: Integer. Range: [0,∞);  Default value: 6

      eta=0.2
       - Step size shrinkage used in updates to prevent overfitting.
       - After each boosting step, you can directly get the weights of new features.
       - The eta parameter actually shrinks the feature weights to make the boosting process more conservative.
       - Optional;  Valid values: Float. Range: [0,1].;  Default value: 0.3

      gamma=4,
        - Minimum loss reduction required to make a further partition on a leaf node of the tree.
        - The larger, the more conservative the algorithm is.
        - Optional;   Valid values: Float. Range: [0,∞).;  Default value: 0

      min_child_weight=6
        - Minimum sum of instance weight (hessian) needed in a child.
        - If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
          the building process gives up further partitioning.
        - In linear regression models, this simply corresponds to a minimum number of instances needed in each node.
        - The larger the algorithm, the more conservative it is.
        - Optional;  Valid values: Float. Range: [0,∞).;  Default value: 1

      subsample=0.8
        - Subsample ratio of the training instance.
        - Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees.
        - This prevents overfitting.
        - Optional;  Valid values: Float. Range: [0,1].;  Default value: 1

      verbosity=0
        - Verbosity of printing messages.
        - Valid values: 0 (silent), 1 (warning), 2 (info), 3 (debug).
        - Optional;  Default value: 1;

      num_round=100
        - The number of rounds to run the training.
        - Required;  Valid values: Integer.


    Note: Hit default transform resource limit

    ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the CreateTransformJob operation: The
    account-level service limit 'ml.m5.large for transform job usage' is 0 Instances, with current utilization of 0 Instances
    and a request delta of 1 Instances. Please use AWS Service Quotas to request an increase for this quota. If AWS Service
    Quotas is not sagemaker-hosted-model-endpoint-demoavailable, contact AWS support to request an increase for this quota.

    -> Service Quotas -> Sagemaker -> ml.m5.large for transform job usage -> increase +1


   -----
   class sagemaker.inputs.TrainingInput(s3_data, distribution=None, compression=None, content_type=None,
                 record_wrapping=None, s3_data_type='S3Prefix',...)

      s3_data (str or PipelineVariable)
        – Defines the location of S3 data to train on.

      distribution (str or PipelineVariable)
        – Valid values: 'FullyReplicated', 'ShardedByS3Key' (default: 'FullyReplicated').

      compression (str or PipelineVariable)
        – Valid values: 'Gzip', None (default: None). This is used only in Pipe input mode.

      content_type (str or PipelineVariable)
        – MIME type of the input data (default: None).

      record_wrapping (str or PipelineVariable)
        – Valid values: ‘RecordIO’ (default: None).

      s3_data_type (str or PipelineVariable)
        – Valid values: 'S3Prefix', 'ManifestFile', 'AugmentedManifestFile'.
        - If 'S3Prefix', s3_data defines a prefix of s3 objects to train on. All objects with s3 keys beginning
          with s3_data will be used to train.
        - If 'ManifestFile' or 'AugmentedManifestFile', then s3_data defines a single S3 manifest file or augmented
          manifest file respectively, listing the S3 data to train on. Both the ManifestFile and AugmentedManifestFile
          formats are described at S3DataSource in the Amazon SageMaker API reference.
   -----
      >>> train_data = sagemaker.inputs.TrainingInput(
      >>>     "s3://{}/{}/train".format(bucket, prefix),
      >>>     distribution="FullyReplicated",
      >>>     content_type="text/csv",
      >>>     s3_data_type="S3Prefix",
   -----

    ----------------
    Code: batch-transform-tumor-prediction


      >>> # # Amazon SageMaker Batch Transform Demo

      >>> # Use SageMaker's XGBoost to train a binary classification model and for a list of tumors in batch file,
      >>> # predict if each is malignant
      >>> #
      >>> # Based on AWS sample located at:
      >>> #  https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker_batch_transform/batch_transform_associate_predictions_with_input
      >>> # ## Setup
      >>> #
      >>> # After installing the SageMaker Python SDK
      >>> # specify:
      >>> #
      >>> # * The SageMaker role arn which has the SageMakerFullAccess policy attached
      >>> # * The S3 bucket to use for training and storing model objects.


      >>> !pip3 install -U sagemaker

      >>> import os
      >>> import boto3
      >>> import sagemaker

      >>> role = sagemaker.get_execution_role()
      >>> sess = sagemaker.Session()
      >>> region = sess.boto_region_name

      >>> bucket = sess.default_bucket()
      >>> prefix = "DEMO-breast-cancer-prediction-xgboost-highlevel"


      >>> # ---
      >>> # ## Data sources
      >>> #
      >>> # > Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA:
      >>> #  University of California, School of Information and Computer Science.
      >>> #
      >>> # ## Data preparation
      >>> #
      >>> # Download the data and save it in a local folder with the name data.csv


      >>> import pandas as pd
      >>> import numpy as np

      >>> s3 = boto3.client("s3")

      >>> filename = "wdbc.csv"
      >>> s3.download_file(
      >>>     f"sagemaker-example-files-prod-{region}", "datasets/tabular/breast_cancer/wdbc.csv", filename
      >>> )
      >>> data = pd.read_csv(filename, header=None)

      >>> # specify columns extracted from wbdc.names
      >>> data.columns = [
      >>>     "id",
      >>>     "diagnosis",
      >>>     "radius_mean",
      >>>     "texture_mean",
      >>>     "perimeter_mean",
      >>>     "area_mean",
      >>>     "smoothness_mean",
      >>>     "compactness_mean",
      >>>     "concavity_mean",
      >>>     "concave points_mean",
      >>>     "symmetry_mean",
      >>>     "fractal_dimension_mean",
      >>>     "radius_se",
      >>>     "texture_se",
      >>>     "perimeter_se",
      >>>     "area_se",
      >>>     "smoothness_se",
      >>>     "compactness_se",
      >>>     "concavity_se",
      >>>     "concave points_se",
      >>>     "symmetry_se",
      >>>     "fractal_dimension_se",
      >>>     "radius_worst",
      >>>     "texture_worst",
      >>>     "perimeter_worst",
      >>>     "area_worst",
      >>>     "smoothness_worst",
      >>>     "compactness_worst",
      >>>     "concavity_worst",
      >>>     "concave points_worst",
      >>>     "symmetry_worst",
      >>>     "fractal_dimension_worst",
      >>> ]

      >>> # save the data
      >>> data.to_csv("data.csv", sep=",", index=False)

      >>> data.sample(8)


      >>> # #### Note:
      >>> # * The first field is an 'id' attribute that we'll remove before batch inference since it is not useful for inference
      >>> # * The second field, 'diagnosis', uses 'M' for Malignant and 'B'for Benign.
      >>> # * There are 30 other numeric features that will be use for training and inferenc.

      >>> # Replace the M/B diagnosis with a 1/0 boolean value.


      >>> data["diagnosis"] = data["diagnosis"].apply(lambda x: ((x == "M")) + 0)
      >>> data.sample(8)


      >>> # Split the data as follows:
      >>> # 80% for training
      >>> # 10% for validation
      >>> # 10% for batch inference job
      >>> #
      >>> # In addition, remove the 'id' field from the training set and validation set as 'id' is not a training feature.
      >>> # Remove the diagnosis attribute for the batch set because this is what we want to predict.


      >>> # data split in three sets, training, validation and batch inference
      >>> rand_split = np.random.rand(len(data))
      >>> train_list = rand_split < 0.8
      >>> val_list = (rand_split >= 0.8) & (rand_split < 0.9)
      >>> batch_list = rand_split >= 0.9

      >>> data_train = data[train_list].drop(["id"], axis=1)
      >>> data_val = data[val_list].drop(["id"], axis=1)
      >>> data_batch = data[batch_list].drop(["diagnosis"], axis=1)
      >>> data_batch_noID = data_batch.drop(["id"], axis=1)


      >>> # Upload the data sets to S3


      >>> train_file = "train_data.csv"
      >>> data_train.to_csv(train_file, index=False, header=False)
      >>> sess.upload_data(train_file, key_prefix="{}/train".format(prefix))

      >>> validation_file = "validation_data.csv"
      >>> data_val.to_csv(validation_file, index=False, header=False)
      >>> sess.upload_data(validation_file, key_prefix="{}/validation".format(prefix))

      >>> batch_file = "batch_data.csv"
      >>> data_batch.to_csv(batch_file, index=False, header=False)
      >>> sess.upload_data(batch_file, key_prefix="{}/batch".format(prefix))

      >>> batch_file_noID = "batch_data_noID.csv"
      >>> data_batch_noID.to_csv(batch_file_noID, index=False, header=False)
      >>> sess.upload_data(batch_file_noID, key_prefix="{}/batch".format(prefix))


      >>> # ---
      >>> #
      >>> # ## Training job and model creation

      >>> # Start the training job using both training set and validation set.
      >>> #
      >>> # The model will output a probability between 0 and 1 which is predicting the probability of a tumor being malignant.


      >>> %%time
      >>> from time import gmtime, strftime

      >>> job_name = "xgb-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
      >>> output_location = "s3://{}/{}/output/{}".format(bucket, prefix, job_name)
      >>> image = sagemaker.image_uris.retrieve(
      >>>     framework="xgboost", region=boto3.Session().region_name, version="1.7-1"
      >>> )

      >>> sm_estimator = sagemaker.estimator.Estimator(
      >>>     image,
      >>>     role,
      >>>     instance_count=1,
      >>>     instance_type="ml.m5.large",
      >>>     volume_size=50,
      >>>     input_mode="File",
      >>>     output_path=output_location,
      >>>     sagemaker_session=sess,
      >>> )

      >>> sm_estimator.set_hyperparameters(
      >>>     objective="binary:logistic",
      >>>     max_depth=5,
      >>>     eta=0.2,
      >>>     gamma=4,
      >>>     min_child_weight=6,
      >>>     subsample=0.8,
      >>>     verbosity=0,
      >>>     num_round=100,
      >>> )

      >>> train_data = sagemaker.inputs.TrainingInput(
      >>>     "s3://{}/{}/train".format(bucket, prefix),
      >>>     distribution="FullyReplicated",
      >>>     content_type="text/csv",
      >>>     s3_data_type="S3Prefix",
      >>> )
      >>> validation_data = sagemaker.inputs.TrainingInput(
      >>>     "s3://{}/{}/validation".format(bucket, prefix),
      >>>     distribution="FullyReplicated",
      >>>     content_type="text/csv",
      >>>     s3_data_type="S3Prefix",
      >>> )
      >>> data_channels = {"train": train_data, "validation": validation_data}

      >>> # Start training by calling the fit method in the estimator
      >>> sm_estimator.fit(inputs=data_channels, logs=True)

      >>> # ---
      >>> #
      >>> # ## Batch Transform
      >>> # Instead of deploying an endpoint and running real-time inference, we'll use SageMaker Batch Transform to run inference on an entire data set in one operation.
      >>> #

      >>> # #### 1. Create a transform job
      >>> #


      >>> %%time

      >>> sm_transformer = sm_estimator.transformer(1, "ml.m5.large")

      >>> # start a transform job
      >>> input_location = "s3://{}/{}/batch/{}".format(
      >>>     bucket, prefix, batch_file_noID
      >>> )  # use input data without ID column
      >>> sm_transformer.transform(input_location, content_type="text/csv", split_type="Line")
      >>> sm_transformer.wait()


      >>> # Check the output of the Batch Transform job. It should show the list of probabilities of tumors being malignant.

      >>> import re

      >>> def get_csv_output_from_s3(s3uri, batch_file):
      >>>     file_name = "{}.out".format(batch_file)
      >>>     match = re.match("s3://([^/]+)/(.*)", "{}/{}".format(s3uri, file_name))
      >>>     output_bucket, output_prefix = match.group(1), match.group(2)
      >>>     s3.download_file(output_bucket, output_prefix, file_name)
      >>>     return pd.read_csv(file_name, sep=",", header=None)


      >>> output_df = get_csv_output_from_s3(sm_transformer.output_path, batch_file_noID)
      >>> output_df.head(8)


      >>> # #### 2. Join the input and the prediction results
      >>> #
      >>> # We can use batch transform to perform a different transform job to join our original data,
      >>> # with our results to get the ID field back.
      >>> #
      >>> # Associate the prediction results with their corresponding input records. We can  use the "input_filter" to exclude
      >>> # the ID column easily and there's no need to have a separate file in S3.
      >>> #
      >>> #  * Set "input_filter" to "$[1:]": indicates that we are excluding column 0 (the 'ID') before processing the
      >>> #   inferences and keeping everything from column 1 to the last column (all the features or predictors)
      >>> #
      >>> # * Set "join_source" to "Input": indicates our desire to join the input data with the inference results
      >>> #
      >>> # * Set "output_filter" to default "$[1:]", indicating that when presenting the output, we only want to keep
      >>> #   column 0 (the 'ID') and the last column (the inference result)


      >>> # content_type / accept and split_type / assemble_with are required to use IO joining feature
      >>> sm_transformer.assemble_with = "Line"
      >>> sm_transformer.accept = "text/csv"

      >>> # start a transform job
      >>> input_location = "s3://{}/{}/batch/{}".format(
      >>>     bucket, prefix, batch_file
      >>> )  # use input data with ID column cause InputFilter will filter it out
      >>> sm_transformer.transform(
      >>>     input_location,
      >>>     split_type="Line",
      >>>     content_type="text/csv",
      >>>     input_filter="$[1:]",
      >>>     join_source="Input",
      >>>     output_filter="$[0,-1]",
      >>> )
      >>> sm_transformer.wait()

      >>> # Check the output of the Batch Transform job in S3. It should show the list of probabilities along with the record ID.

      >>> output_df = get_csv_output_from_s3(sm_transformer.output_path, batch_file)
      >>> output_df.head(8)


      >>> # ## Clean up
      >>> # In the AWS console, we can see that a model has been created, S3 buckets, and batch transform jobs, however
      >>> # no SageMaker endpoint has been created.
      >>> #
      >>> # To avoid unnecessary charges, be sure to delete:
      >>> # - S3 buckets
      >>> # - Model
      >>> # - Jupter notebook

    ----------------

  Clean up;

     Sagemaker -> models -> select <model> -> delete

     S3 -> <bucket> -> empty -> delete


     SageMaker -> Studio -> JupyterLab -> kernelks -> shut down all

     SageMaker -> Studio -> JupyterLab -> running instances -> select "MyJupterLab" -> stop
        -> click on "MyJupyterLab" instance -> <upper right - 3 dots> -> Delect Space


------------------------------------------------------

