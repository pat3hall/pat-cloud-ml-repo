------------------------------------------------------
4.5 Training s SageMaker Model Using a Training Script

  Saved files:
    SageMaker entrypoint training script
      train.py
    original python jupyter notebook:
      sklean_byom.ipynb
    completed python jupyter notebook:
      sklean_byom_output.ipynb
    extracted python code from jupyter notebook:
      sklean_byom.py

  Using Scikit-learn with the SageMaker Python SDK
    https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html

  Algorithm Implementation options
    built-in algorithm.
      - no code required
      - requires algorithm with training data, hyperparameters, and computing resources.
    script mode in supported framework
      - Develop a custom Python script
      - in a supported framework, like Psyche learn, TensorFlow, pyarch, or MXNet.
      - leverage the additional Python libraries that are preloaded with these frameworks for
        training an algorithm.
    custom docker image,
      - requires docker expertise
      - if your use case is not addressed by previous two options.
      - the Docker image must be uploaded to Amazon ECR before you can start training the model.

  Demo Example
    - IRIS dataset, which represents instances of various plant species.
    - multi-class classification model that will predict the plant species based on the sepal and
      the petal length and width.
    - using random forest regressor in this demo.
    - pre-processing data and convert the class labels from string to integer.
    - data is split into training and validation data in a 80/20 ratio and uploaded to S3 bucket.
    - need  actual training script along with the training data.



   SageMaker Notebook instance -> Jupyter -> SageMaker Examples -> SageMaker Script Mode -> sklearn_byom.ipynb -> Use

   -> downloads:
       sklearn_2024-12-13/sklearn_pyom.ipynb
       sklearn_2024-12-13/sklearn_pyom_outputs.ipynb
       sklearn_2024-12-13/train.py


  Traning script:
    - A typical training script loads data from the input channels, configures training with hyperparameters,
      trains a model, and saves a model to model_dir so that it can be hosted later.
    - Hyperparameters are passed to your script as arguments and can be retrieved with an argparse.ArgumentParser instance.

    Training Script env variables:
      SM_MODEL_DIR:
        - the path to the directory to write model artifacts to.
      SM_OUTPUT_DATA_DIR:
        - path to write output artifacts to. Output artifacts
      SM_CHANNEL_TRAIN:
        - A string representing the path to the directory containing data in the ‘train’ channel
      SM_CHANNEL_TEST:
        - Same as above, but for the ‘test’ channel.


    Code:  Train a SKLearn Model using Script Mode

      >>> #
      >>> # The aim of this notebook is to demonstrate how to train and deploy a scikit-learn model in Amazon SageMaker. The
      >>> # method used is called Script Mode, in which we write a script to train our model and submit it to the SageMaker
      >>> # Python SDK. For more information, feel free to read [Using Scikit-learn with the SageMaker Python SDK]
      >>> # (https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html).
      >>> #
      >>> # ## Runtime
      >>> # This notebook takes approximately 15 minutes to run.
      >>> #
      >>> # ## Contents
      >>> # 1. [Download data](#Download-data)
      >>> # 2. [Prepare data](#Prepare-data)
      >>> # 3. [Train model](#Train-model)
      >>> # 4. [Deploy and test endpoint](#Deploy-and-test-endpoint)
      >>> # 5. [Cleanup](#Cleanup)

      >>> # ## Download data
      >>> # Download the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris), which is the data used to
      >>> # trained the model in this demo.

      >>> get_ipython().system('pip install -U sagemaker')


      >>> import boto3
      >>> import pandas as pd
      >>> import numpy as np

      >>> s3 = boto3.client("s3")
      >>> s3.download_file(
      >>>     f"sagemaker-example-files-prod-{boto3.session.Session().region_name}",
      >>>     "datasets/tabular/iris/iris.data",
      >>>     "iris.data",
      >>> )

      >>> df = pd.read_csv(
      >>>     "iris.data", header=None, names=["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
      >>> )
      >>> df.head()


      >>> # ## Prepare data
      >>> # Next, we prepare the data for training by first converting the labels from string to integers. Then we split
      >>> # the data into a train dataset (80% of the data) and test dataset (the remaining 20% of the data) before saving
      >>> # them into CSV files. Then, these files are uploaded to S3 where the SageMaker SDK can access and use them to
      >>> # train the model.

      >>> # Convert the three classes from strings to integers in {0,1,2}
      >>> df["class_cat"] = df["class"].astype("category").cat.codes
      >>> categories_map = dict(enumerate(df["class"].astype("category").cat.categories))
      >>> print(categories_map)
      >>> df.head()


      >>> df_rand= df.sample(frac=1)
      >>> df_rand.head()


      >>> # Split the data into 80-20 train-test split
      >>> num_samples = df_rand.shape[0]
      >>> split = round(num_samples * 0.8)
      >>> train = df_rand.iloc[:split, :]
      >>> test = df_rand.iloc[split:, :]
      >>> print("{} train, {} test".format(split, num_samples - split))


      >>> # Write train and test CSV files
      >>> train.to_csv("train.csv", index=False)
      >>> test.to_csv("test.csv", index=False)


      >>> # Create a sagemaker session to upload data to S3
      >>> import sagemaker

      >>> sagemaker_session = sagemaker.Session()

      >>> # Upload data to default S3 bucket
      >>> prefix = "DEMO-sklearn-iris"
      >>> training_input_path = sagemaker_session.upload_data("train.csv", key_prefix=prefix + "/training")


      >>> # ## Train model
      >>> # The model is trained using the SageMaker SDK's Estimator class. Firstly, get the execution role for training.
      >>> # This role allows us to access the S3 bucket in the last step, where the train and test data set is located.

      >>> # Use the current execution role for training. It needs access to S3
      >>> role = sagemaker.get_execution_role()
      >>> print(role)


      >>> # Then, it is time to define the SageMaker SDK Estimator class. We use an Estimator class specifically desgined to train
      >>> # scikit-learn models called `SKLearn`. In this estimator, we define the following parameters:
      >>> # 1. The script that we want to use to train the model (i.e. `entry_point`). This is the heart of the Script Mode method.
      >>> # Additionally, set the `script_mode` parameter to `True`.
      >>> # 2. The role which allows us access to the S3 bucket containing the train and test data set (i.e. `role`)
      >>> # 3. How many instances we want to use in training (i.e. `instance_count`) and what type of instance we want to use in
      >>> # training (i.e. `instance_type`)
      >>> # 4. Which version of scikit-learn to use (i.e. `framework_version`)
      >>> # 5. Training hyperparameters (i.e. `hyperparameters`)
      >>> #
      >>> # After setting these parameters, the `fit` function is invoked to train the model.

      >>> # Docs: https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html

      >>> from sagemaker.sklearn import SKLearn

      >>> sk_estimator = SKLearn(
      >>>     entry_point="train.py",
      >>>     role=role,
      >>>     instance_count=1,
      >>>     instance_type="ml.c5.xlarge",
      >>>     py_version="py3",
      >>>     framework_version="1.2-1",
      >>>     script_mode=True,
      >>>     hyperparameters={"estimators": 20},
      >>> )

      >>> # Train the estimator
      >>> sk_estimator.fit({"train": training_input_path})


      >>> # ## Deploy and test endpoint
      >>> # After training the model, it is time to deploy it as an endpoint. To do so, we invoke the `deploy` function within
      >>> # the scikit-learn estimator. As shown in the code below, one can define the number of instances (i.e. `initial_instance_count`)
      >>> # and instance type (i.e. `instance_type`) used to deploy the model.

      >>> import time

      >>> sk_endpoint_name = "sklearn-rf-model" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
      >>> sk_predictor = sk_estimator.deploy(
      >>>     initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=sk_endpoint_name
      >>> )


      >>> # After the endpoint has been completely deployed, it can be invoked using the [SageMaker Runtime Client]
      >>> # (https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html)
      >>> # (which is the method used in the code cell below) or [Scikit Learn Predictor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-predictor). If you plan to use the latter method, make sure to use a [Serializer](https://sagemaker.readthedocs.io/en/stable/api/inference/serializers.html) to serialize your data properly.

      >>> import json

      >>> client = sagemaker_session.sagemaker_runtime_client

      >>> request_body = {"Input": [[9.0, 3571, 1976, 0.525]]}
      >>> data = json.loads(json.dumps(request_body))
      >>> payload = json.dumps(data)

      >>> response = client.invoke_endpoint(
      >>>     EndpointName=sk_endpoint_name, ContentType="application/json", Body=payload
      >>> )

      >>> result = json.loads(response["Body"].read().decode())["Output"]
      >>> print("Predicted class category {} ({})".format(result, categories_map[result]))


      >>> # ## Cleanup
      >>> # If the model and endpoint are no longer in use, they should be deleted to save costs and free up resources.

      >>> sk_predictor.delete_model()
      >>> sk_predictor.delete_endpoint()


    Code:  Entrypoint script 'train.py' called by above SageMaker script mode script

      >>> import argparse, os
      >>> import boto3
      >>> import json
      >>> import pandas as pd
      >>> import numpy as np
      >>> from sklearn.model_selection import train_test_split
      >>> from sklearn.preprocessing import StandardScaler
      >>> from sklearn.ensemble import RandomForestRegressor
      >>> from sklearn import metrics
      >>> import joblib

      >>> if __name__ == "__main__":

      >>>     # Pass in environment variables and hyperparameters
      >>>     parser = argparse.ArgumentParser()

      >>>     # Hyperparameters
      >>>     parser.add_argument("--estimators", type=int, default=15)

      >>>     # sm_model_dir: model artifacts stored here after training
      >>>     parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
      >>>     parser.add_argument("--model_dir", type=str)
      >>>     parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

      >>>     args, _ = parser.parse_known_args()
      >>>     estimators = args.estimators
      >>>     model_dir = args.model_dir
      >>>     sm_model_dir = args.sm_model_dir
      >>>     training_dir = args.train

      >>>     # Read in data
      >>>     df = pd.read_csv(training_dir + "/train.csv", sep=",")

      >>>     # Preprocess data
      >>>     X = df.drop(["class", "class_cat"], axis=1)
      >>>     y = df["class_cat"]
      >>>     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      >>>     sc = StandardScaler()
      >>>     X_train = sc.fit_transform(X_train)
      >>>     X_test = sc.transform(X_test)

      >>>     # Build model
      >>>     regressor = RandomForestRegressor(n_estimators=estimators)
      >>>     regressor.fit(X_train, y_train)
      >>>     y_pred = regressor.predict(X_test)

      >>>     # Save model
      >>>     joblib.dump(regressor, os.path.join(args.sm_model_dir, "model.joblib"))

      >>> # Model serving
      >>> # INFERENCE
      >>> # SageMaker uses four functions to load the model and use it for inference: model_fn, input_fn, output_fn, and predict_fn

      >>> """
      >>> Deserialize fitted model
      >>> """
      >>> def model_fn(model_dir):
      >>>     model = joblib.load(os.path.join(model_dir, "model.joblib"))
      >>>     return model

      >>> """
      >>> input_fn
      >>>     request_body: The body of the request sent to the model.
      >>>     request_content_type: (string) specifies the format/variable type of the request
      >>> """
      >>> def input_fn(request_body, request_content_type):
      >>>     if request_content_type == "application/json":
      >>>         request_body = json.loads(request_body)
      >>>         inpVar = request_body["Input"]
      >>>         return inpVar
      >>>     else:
      >>>         raise ValueError("This model only supports application/json input")

      >>> """
      >>> predict_fn
      >>>     input_data: returned array from input_fn above
      >>>     model (sklearn model) returned model loaded from model_fn above
      >>> """
      >>> def predict_fn(input_data, model):
      >>>     return model.predict(input_data)

      >>> """
      >>> output_fn
      >>>     prediction: the returned value from predict_fn above
      >>>     content_type: the content type the endpoint expects to be returned. Ex: JSON, string
      >>> """
      >>> def output_fn(prediction, content_type):
      >>>     res = int(prediction[0])
      >>>     respJSON = {"Output": res}
      >>>     return respJSON


