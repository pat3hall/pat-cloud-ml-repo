------------------------------------------------------
4.11 Linear Regression Performed Using Amazon SageMaker

Saved files:
    completed python jupyter notebook:
      train.ipynb
    extracted python code from jupyter notebook:
      train.py
    html view from completed jupyter notebook:
      train.html

About this lab

Imagine you are the data engineer at your company and your company just selected AWS as the preferred cloud provider.
You have been asked to train an ML model using linear learner algorithm. In this lab, you will fetch the iris data and
use that as the input dataset. Once the data is split, the data is uploaded to S3 bucket. Then the Sagemaker estimator
is configured before initiating the training process.

Learning objectives
  - Launch SageMaker Notebook
  - Install dependencies and import the libraries
  - Download the data and upload them to S3 bucket
  - Set up training and validation data
  - Train the model

--------------------------
Solution
Launch SageMaker Notebook

    To avoid issues with the lab, open a new Incognito or Private browser window to log in to the lab. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region. If you are prompted to select a kernel, please choose conda_tensorflow2_p310.
    In the search bar, type "SageMaker" to search for the SageMaker service. Click on the Amazon SageMaker result to go directly to the SageMaker service.
    Click on the Notebooks link under the Applications and IDEs section to view the notebook provided by the lab.
    Check if the notebook is marked as** InService.** If so, click the Open Jupyter link under** Actions.**
    Click on the train.ipnyb file.

Load the Dataset and Split the Data

Note: If this is your first time running a notebook, each cell contains Python commands you can run independently. Also, a * inside the square braces indicates the code is running, and you will see a number once the execution is complete.

    Click the first cell that installs boto3 and sagemaker. Use the Run button at the top to execute the code.
    Click the next cell and the Run button to import the required Python libraries, initialize the SageMaker session and define the output bucket. This cell lists all the libraries that will be imported from Pandas and sklearn. It may take a few minutes to complete the operation.
    Copy the following code snippet and paste into the next cell, click Run to perform this operation. This cell fetches the IAM role using the get_execution_role function.

    role = get_execution_role()

Download the data and upload them to the S3 bucket

    Select the next cell and click Run to execute. This cell fetches the iris dataset from the sklearn library. After loading the dataset, we create feature variables (X) and target variables (Y).

    Copy the following code snippet and paste into the next cell. Select the cell and click Run to split the data for training and testing purposes.

    train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

    Select the next cell and click Run to convert the training and validation data to CSV format.

    Select the next cell and click Run to upload the training CSV file to the S3 bucket using the upload_file function.

    Copy the following code snippet and paste into the next cell. Click Run to execute the code that will upload the validation.csv file.

    s3.upload_file('validation.csv', output_bucket, f'{output_prefix}/validation/validation.csv')

Setup Training and Validation Data

    Select the next cell and click Run to initialize the path from which training data and validation data will be read. These values will be used to create the input parameter for the estimator object.
    Select the next cell and click Run. This code uses the TrainingInput function and creates the train_data input parameter. Please pay attention; we are passing the path from which the training data will be read.
    Copy the following code snippet and paste into the next cell, and click Run to create an input parameter for validation_data,

    validation_data = sagemaker.inputs.TrainingInput(
        s3_validation_data,
        distribution="FullyReplicated",
        content_type="text/csv",
        s3_data_type="S3Prefix",
        record_wrapping=None,
        compression=None,
    )

Fetch the Algorithm and Train the Model

In this section, we will see how to train the linear learner model.

    Select the first cell and click Run. This cell fetches the linear-learner algorithm specific to the region.
    Select the next cell and click Run to initialize the Estimator object. Please make sure you choose the correct instance type and instance count.
    Copy the following code snippet and paste into the next cell, click Run to configure the hyperparameters manually.

    linear.set_hyperparameters(
        feature_dim=4,  # Adjust this to match your feature dimension
        predictor_type='regressor',  # Use 'classifier' for classification
        mini_batch_size=20
    )

    Select the final cell and click Run. This cell uses the fit function to initiate the training process.

Once the training process starts, you may switch to SageMaker console and monitor the training process.
--------------------------

    code:  Linear Learner regressor lab

      >>> # Linear Regression Performed Using Amazon SageMaker

      >>> # # Introduction
      >>> #
      >>> # In this lab, you will learn how to import the iris dataset, split it into training and validation data, upload
      >>> # them to S3 bucket, fetch the linear learner algorithm, initialize the estimator object, set the hyperparameters
      >>> # and train the model.

      >>> # # How to Use This Lab
      >>> #
      >>> # Most of the code is provided for you in this lab as our solution to the tasks presented. Some of the cells are left
      >>> # empty with a #TODO header and its your turn to fill in the empty code. You can always use our lab guide if you are stuck.

      >>> # # 1) Install dependencies and import the required libraries

      >>> # Install Sagemaker
      >>> get_ipython().system('pip install boto3 sagemaker')


      >>> # 1. We will use the iris dataset as our input data.
      >>> # 2. The S3 bucket that you want to use for training data must be within the same region as the Notebook Instance.
      >>> # 3. The IAM role is used to provide training and hosting access to your data. See the documentation for how to create these.
      >>> # Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp
      >>> # with an appropriate full IAM role arn string(s).

      >>> import sagemaker
      >>> import boto3
      >>> from sagemaker import get_execution_role
      >>> from sagemaker.inputs import TrainingInput
      >>> import pandas as pd
      >>> from sklearn.datasets import load_iris
      >>> from sklearn.model_selection import train_test_split

      >>> # Initialize the SageMaker session
      >>> sagemaker_session = sagemaker.Session()

      >>> # Define the S3 bucket and prefix to store data
      >>> output_bucket = sagemaker.Session().default_bucket()
      >>> output_prefix = 'sagemaker/linear-learner'


      >>> #TODO: Fetch the IAM role using the get_execution_role function and assign the value to a variable `role`
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


      >>> data.head()


      >>> #TODO: Use the `train_test_split` function and split the data in a 80 - 20 ratio.
      >>> #TODO: Assign the values to variables `train_data` and `validation_data`
      >>> train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)


      >>> # Save to CSV
      >>> train_data.to_csv('train.csv', index=False, header=False)
      >>> validation_data.to_csv('validation.csv', index=False, header=False)


      >>> # Let's use the upload_file function and upload the .csv files to the S3 buckets

      >>> # Upload data to S3
      >>> s3 = boto3.client('s3')
      >>> s3.upload_file('train.csv', output_bucket, f'{output_prefix}/train/train.csv')


      >>> #TODO: Using the strategy we followed to upload the training data, as shown above, please upload the validation
      >>> # data to the output bucket.
      >>> s3.upload_file('validation.csv', output_bucket, f'{output_prefix}/validation/validation.csv')


      >>> # # 3) Set up training and validation data

      >>> # Create three separate variables that are dynamically constructed, which will be used as one of the input
      >>> # parameters while generating training inputs.

      >>> # creating the inputs for the fit() function with the training and validation location
      >>> s3_train_data = f"s3://{output_bucket}/{output_prefix}/train"
      >>> print(f"training files will be taken from: {s3_train_data}")
      >>> s3_validation_data = f"s3://{output_bucket}/{output_prefix}/validation"
      >>> print(f"validation files will be taken from: {s3_validation_data}")
      >>> output_location = f"s3://{output_bucket}/{output_prefix}/output"
      >>> print(f"training artifacts output location: {output_location}")


      >>> # Let's create the sagemaker.session.s3_input objects from our data channels. Note that we are using the
      >>> # content_type as text/csv. We use two channels here, one for training and the second for validation.

      >>> # generating the session.s3_input() format for fit() accepted by the sdk
      >>> train_data = sagemaker.inputs.TrainingInput(
      >>>     s3_train_data,
      >>>     distribution="FullyReplicated",
      >>>     content_type="text/csv",
      >>>     s3_data_type="S3Prefix",
      >>>     record_wrapping=None,
      >>>     compression=None,
      >>> )


      >>> #TODO: Following the same strategy shown above, please set up a training input for validation data.
      >>> #TODO: Name it as `validation_data`
      >>> validation_data = sagemaker.inputs.TrainingInput(
      >>>     s3_validation_data,
      >>>     distribution="FullyReplicated",
      >>>     content_type="text/csv",
      >>>     s3_data_type="S3Prefix",
      >>>     record_wrapping=None,
      >>>     compression=None,
      >>> )


      >>> # # 4) Fetch the algorithm and train the model

      >>> # Let's retrieve the image for the Linear Learner Algorithm according to the region.

      >>> # Fetch the linear learner image according to the region
      >>> from sagemaker.image_uris import retrieve

      >>> container = retrieve("linear-learner", boto3.Session().region_name, version="1")
      >>> print(container)
      >>> deploy_amt_model = True


      >>> # Then, we create an estimator from the SageMaker Python SDK using the Linear Learner container image
      >>> # and set the training parameters.

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


      >>> # The hyperparameters are manually configured

      >>> # TODO: Use the set_hyperparameters function and set the following hyperparameters on linear learner
      >>> # feature_dim=4, predictor_type='regressor', mini_batch_size=20
      >>> linear.set_hyperparameters(
      >>>     feature_dim=4,  # Adjust this to match your feature dimension
      >>>     predictor_type='regressor',  # Use 'classifier' for classification
      >>>     mini_batch_size=20
      >>> )


      >>> # 1. The following cell will train the algorithm. Training the algorithm involves a few steps. First, the instances
      >>> #    that we requested while creating the Estimator classes are provisioned and set up with the appropriate libraries.
      >>> #    Then, the data from our channels is downloaded into the instance. Once this is done, the training job begins.
      >>> #    The provisioning and data downloading will take time, depending on the size of the data. Therefore, it might be
      >>> #    a few minutes before we start getting data logs for our training jobs.
      >>> # 2. The log will print the objective metric details.
      >>> # 3. The training time takes between 4 and 6 minutes.

      >>> %%time
      >>> linear.fit(inputs={"train": train_data, "validation": validation_data}, job_name=job_name)


