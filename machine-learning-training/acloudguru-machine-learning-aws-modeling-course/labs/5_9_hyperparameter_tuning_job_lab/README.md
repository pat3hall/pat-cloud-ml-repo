------------------------------------------------------
5.9 Hyperparameter Tuning Job Created Using Amazon SageMaker


Saved files:
    completed python jupyter notebook:
      tune.ipynb
    extracted python code from jupyter notebook:
      tune.py
    html view from completed jupyter notebook:
      tune.html


About this lab

Imagine you are a data engineer, and your company just selected AWS as the preferred cloud provider. You have been
asked to find the optimal set of hyperparameters for the linear learner model by leveraging SageMaker's Automatic
Model Tuning (AMT). In this lab, you will fetch the iris data and use that as the input dataset. Once the data is
split, the data is uploaded to an S3 bucket. Then the Sagemaker estimator is configured to run four tuning jobs,
and will identify the job that minimizes the objective metric.

Learning objectives
 - Launch SageMaker Notebook
 - Install dependencies and import the libraries
 - Download the data and upload it to an S3 bucket
 - Set up training and validation data
 - Initialize the estimator
 - Tune hyperparameters

-------------------------
Solution
Launch SageMaker Notebook

    To avoid issues with the lab, open a new Incognito or Private browser window to log in to the lab. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1, N. Virginia, region. (If you are prompted to select a kernel, choose conda_tensorflow2_p310.)
    In the search bar, enter SageMaker then click on Amazon SageMaker.
    On the left, under the Applications and IDEs section, click on Notebooks.
    Ensure the Notebook is marked as InService (if it's Pending wait for a 30 seconds or a minute), then under Actions, click the Open Jupyter link.
    Click on the tune.ipnyb file.

Load the Dataset and Split the Data

Each cell contains Python commands you can run independently.

    Click the first cell that installs boto3 and sagemaker. Use the Run button at the top to execute the code. A * inside the square braces indicates the code is running, and you will see a number once the execution is complete.
    Run the next cell, which imports the required Python libraries, initializes the sagemaker session, and defines the output bucket. This cell indicates all the libraries you will import from pandas and sklearn. It may take a few minutes to complete the operation.
    The next cell fetches the IAM role using the get_execution_role function. Paste in the following code snippet, ensuring you overwrite the existing comments, and Run the cell to perform this operation.

    role = get_execution_role()

Download the Data and Upload It to the S3 Bucket

    The next cell fetches the iris dataset from the sklearn library. After loading the dataset, you create a feature variable (X), and a target variable (y). Click Run to execute this cell.
    Now that the data is imported, it must be split for training and testing purposes. Use the following code snippet, and Run the cell to perform this operation.

    train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

    Run the next cell converts the training and validation data to CSV format.
    Then, you will upload the training CSV file to the S3 bucket using the upload_file function. (Run this cell.)
    Use the following code snippet, and run it in the next cell to upload the validation.csv file.

    s3.upload_file('validation.csv', output_bucket, f'{output_prefix}/validation/validation.csv')

Set Up Training and Validation Data

    Run the first cell in this section, which initializes the path from which training data and validation data will be read. These values will be used to create the input parameter for the estimator object.
    Then, use the TrainingInput function, and create the train_data input parameter. This passes the path from which the training data will be read.
    Use the following code snippet and create an input parameter for validation_data,

    validation_data = sagemaker.inputs.TrainingInput(
        s3_validation_data,
        distribution="FullyReplicated",
        content_type="text/csv",
        s3_data_type="S3Prefix",
        record_wrapping=None,
        compression=None,
    )

Fetch the Algorithm and Initialize the Estimator

    In this section, you will see how to configure the estimator object. The first cell under this category fetches the linear-learner algorithm specific to the region. Run this cell to fetch the algorithm.
    The next cell initializes the Estimator object. It is important to ensure you choose the correct instance type and instance count.

Define the Hyperparameter Ranges and Tune the Model

    Use the following code snippet to configure the hyperparameters manually.

    linear.set_hyperparameters(
        feature_dim=4,  # Adjust this to match your feature dimension
        predictor_type='regressor',  # Use 'classifier' for classification
        mini_batch_size=20
    )

    Then set the hyperparameter ranges for learning_rate and wd (weight decay).

    # Define the hyperparameter ranges
    hyperparameter_ranges = {
        'learning_rate': ContinuousParameter(0.01, 0.2),
        'wd': ContinuousParameter(0.0, 0.1)
    }

    Now create the HyperparameterTuner object. It is important to set the variables max_jobs and max_parallel_jobs appropriately for your capacity. The goal is to minimize rmse, or Root Mean Square Error, to improve accuracy.
    Initiate the tuning process by invoking the fit function.

    # Launch the hyperparameter tuning job
    tuner.fit({'train': train_data, 'validation': validation_data})

    You will get a couple No finished training job messages, which is expected, as no training was done before tuning.
    Go back to SageMaker in the console, on the left scroll down and expand Training, then click Hyperparameter tuning jobs to monitor the tuning.
-------------------------


    code: hyperparameter Tuning Job Created Using Amazon SageMaker

      >>> # # Introduction
      >>> #
      >>> # In this lab, you will learn how to import the iris dataset, split it into training and validation data, upload them to the
      >>> # S3 bucket, fetch the linear learner algorithm, initialize the estimator object, and automatically tune the hyperparameters
      >>> # using Amazon SageMaker's Automatic Model Tuning (AMT).

      >>> # # How to Use This Lab
      >>> #
      >>> # Most of the code is provided for you in this lab as our solution to the tasks presented. Some of the cells are left empty
      >>> # with a #TODO header, and it's your turn to fill in the empty code. You can always use our lab guide if you are stuck.

      >>> # # 1) Install dependencies and import the required libraries

      >>> # Install Sagemaker
      >>> !pip install boto3 sagemaker


      >>> # 1. We will use the iris dataset as our input data.
      >>> # 2. The S3 bucket you want to use for training data must be within the same region as the Notebook Instance.
      >>> # 3. The IAM role is used to provide training and hosting access to your data. See the documentation for how to create these.
      >>> #    Note that if more than one role is required for notebook instances, training, and/or hosting, please replace the boto
      >>> #    regexp with an appropriate full IAM role arn string(s).

      >>> import sagemaker
      >>> import boto3
      >>> from sagemaker import get_execution_role
      >>> from sagemaker.inputs import TrainingInput
      >>> import pandas as pd
      >>> from sklearn.datasets import load_iris
      >>> from sklearn.model_selection import train_test_split
      >>> from sagemaker.tuner import HyperparameterTuner, ContinuousParameter

      >>> # Initialize the SageMaker session
      >>> sagemaker_session = sagemaker.Session()

      >>> # Define the S3 bucket and prefix to store data
      >>> output_bucket = sagemaker.Session().default_bucket()
      >>> output_prefix = 'sagemaker/linear-learner'


      >>> #TODO: Fetch the IAM role using the get_execution_role function and assign the value to a variable `role.`
      >>> role = get_execution_role()


      >>> # # 2) Download the data and upload them to S3 bucket

      >>> # 1. load_iris function is used to download the input data
      >>> # 2. The data is split into training and validation data in the ratio of 80 - 20
      >>> # 3. The data is saved under 'train.csv' and 'validation.csv'

      >>> # Load the Iris dataset
      >>> iris = load_iris()
      >>> X = iris.data
      >>> y = iris.target

      >>> # Convert to DataFrame for easier manipulation
      >>> data = pd.DataFrame(X, columns=iris.feature_names)
      >>> data['target'] = y


      >>> iris.feature_names


      >>> data.head()


      >>> #TODO: Use the `train_test_split` function and split the data in an 80 - 20 ratio.
      >>> #TODO: Assign the values to variables `train_data` and `validation_data`.
      >>> train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)


      >>> # Save to CSV
      >>> train_data.to_csv('train.csv', index=False, header=False)
      >>> validation_data.to_csv('validation.csv', index=False, header=False)


      >>> # Let's use the upload_file function and upload the .csv files to the S3 buckets.

      >>> # Upload data to S3
      >>> s3 = boto3.client('s3')
      >>> s3.upload_file('train.csv', output_bucket, f'{output_prefix}/train/train.csv')


      >>> #TODO: Using the strategy we followed to upload the training data, as shown above, please upload the validation data to
      >>> # the output bucket.
      >>> s3.upload_file('validation.csv', output_bucket, f'{output_prefix}/validation/validation.csv')


      >>> # # 3) Set up training and validation data

      >>> # Create three separate variables that are dynamically constructed, which will be used as one of the input parameters
      >>> # while generating training inputs.

      >>> # creating the inputs for the fit() function with the training and validation location
      >>> s3_train_data = f"s3://{output_bucket}/{output_prefix}/train"
      >>> print(f"training files will be taken from: {s3_train_data}")
      >>> s3_validation_data = f"s3://{output_bucket}/{output_prefix}/validation"
      >>> print(f"validation files will be taken from: {s3_validation_data}")
      >>> output_location = f"s3://{output_bucket}/{output_prefix}/output"
      >>> print(f"training artifacts output location: {output_location}")


      >>> # Let's create the sagemaker.session.s3_input objects from our data channels. Note that we are using the content_type as
      >>> # text/csv. We use two channels here, one for training and the second for validation.

      >>> # generating the session.s3_input() format for fit() accepted by the sdk
      >>> train_data = sagemaker.inputs.TrainingInput(
      >>>     s3_train_data,
      >>>     distribution="FullyReplicated",
      >>>     content_type="text/csv",
      >>>     s3_data_type="S3Prefix",
      >>>     record_wrapping=None,
      >>>     compression=None,
      >>> )


      >>> #TODO: Following the above strategy, please set up a training input for validation data.
      >>> #TODO: Name it as `validation_data`.
      >>> validation_data = sagemaker.inputs.TrainingInput(
      >>>     s3_validation_data,
      >>>     distribution="FullyReplicated",
      >>>     content_type="text/csv",
      >>>     s3_data_type="S3Prefix",
      >>>     record_wrapping=None,
      >>>     compression=None,
      >>> )

      >>> # # 4) Fetch the algorithm and initialize estimator

      >>> # Let's retrieve the image for the Linear Learner Algorithm according to the region.

      >>> # Fetch the linear learner image according to the region
      >>> from sagemaker.image_uris import retrieve

      >>> container = retrieve("linear-learner", boto3.Session().region_name, version="1")
      >>> print(container)
      >>> deploy_amt_model = True


      >>> # Then, we create an estimator from the SageMaker Python SDK using the Linear Learner container image and set the
      >>> # training parameters.

      >>> %%time
      >>> import boto3
      >>> import sagemaker
      >>> from time import gmtime, strftime

      >>> sess = sagemaker.Session()

      >>> job_name = "linear-learner-iris-regression-" + strftime("%Y%m%d-%H-%M-%S", gmtime())
      >>> print("Training job", job_name)

      >>> linear = sagemaker.estimator.Estimator(
      >>>     container,
      >>>     role,
      >>>     instance_count=1,
      >>>     instance_type="ml.m5.large",
      >>>     output_path=output_location,
      >>>     sagemaker_session=sagemaker_session,
      >>> )

      >>> # # 5) Define hyperparameter ranges and invoke tuning job

      >>> # Set the initial values for the hyperparameters.

      >>> # TODO: Use the set_hyperparameters function and set the initial hyperparameters on linear learner
      >>> # feature_dim=4, predictor_type='regressor', mini_batch_size=20
      >>> linear.set_hyperparameters(
      >>>     feature_dim=4,  # Adjust this to match your feature dimension
      >>>     predictor_type='regressor',  # Use 'classifier' for classification
      >>>     mini_batch_size=20
      >>> )


      >>> # Lets use the Continous parameter range and define the values for `learning rate` and `wd` (weight decay - L2 regularization).

      >>> #TODO: Define the hyperparameter ranges
      >>> #1. 'learning_rate': ContinuousParameter(0.01, 0.2)
      >>> #2. 'wd': ContinuousParameter(0.0, 0.1)
      >>> # Define the hyperparameter ranges
      >>> hyperparameter_ranges = {
      >>>     'learning_rate': ContinuousParameter(0.01, 0.2),
      >>>     'wd': ContinuousParameter(0.0, 0.1)
      >>> }


      >>> # 1. Instead of manually configuring our hyperparameter values and training with SageMaker Training, we'll use Amazon
      >>> # SageMaker Automatic Model Tuning.
      >>> # 2. The code sample below shows you how to use the HyperParameterTuner. It accepts the hyperparameter ranges we set previously.
      >>> # 3. Based on your capacity, you can adjust the `max_jobs` and `max_parallel_jobs.`
      >>> # 4. The goal of the tuning job is to minimize `rmse.`
      >>> # 5. The tuning job will take 8 to 10 minutes to complete.

      >>> # Create a HyperparameterTuner object
      >>> tuner = HyperparameterTuner(
      >>>     estimator=linear,
      >>>     objective_metric_name='validation:rmse',
      >>>     hyperparameter_ranges=hyperparameter_ranges,
      >>>     metric_definitions=[
      >>>         {'Name': 'validation:rmse', 'Regex': 'validation rmse=([0-9\\.]+)'}
      >>>     ],
      >>>     max_jobs=4,
      >>>     max_parallel_jobs=2,
      >>>     objective_type='Minimize'
      >>> )


      >>> #TODO: Initiate the tuner job by invoking the fit function.
      >>> #2. Pass the train_data and validation_data as input parameters.
      >>> # Launch the hyperparameter tuning job
      >>> tuner.fit({'train': train_data, 'validation': validation_data})


      >>> tuner_name = tuner.describe()['HyperParameterTuningJobName']
      >>> print(f'tuning job submitted: {tuner_name}.')


      >>> tuner.best_training_job


      >>> # Retrieve analytics object
      >>> #tuner_analytics = tuner.analytics()

      >>> # Look at summary of associated training jobs
      >>> #tuner_analytics_dataframe = tuner_analytics.dataframe()



