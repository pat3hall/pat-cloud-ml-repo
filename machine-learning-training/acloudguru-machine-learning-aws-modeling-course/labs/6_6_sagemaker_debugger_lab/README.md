------------------------------------------------------
6.6 Training Reports Utilized in SageMaker Debugger to Improve Your Models

Saved files:
    completed python jupyter notebook:
      eval.ipynb
    extracted python code from jupyter notebook:
      eval.py
    html view from completed jupyter notebook:
      eval.html
    SageMaker Debugger report and notebook
      profiler-report.html
      profiler-report.ipynb
    SageMaker Profiler report and notebook
      xgboost_report.html
      xgboost_report.ipynb


About this lab

Imagine you are the data engineer at your company, and your company has just selected AWS as the
preferred cloud provider. You have been given a dataset to predict if an individual makes more than
$50K in salary. As part of the modeling process, you have been asked to generate a summary of the model
training evaluation results, insights into the model performance, and interactive graphs. In this lab,
you will fetch the census data and use that as the input dataset. Once the data is split, the data is
uploaded to the S3 bucket. Then, the Sagemaker estimator is configured with the debugger hook and
Sagemaker built-in rules to generate performance metric reports.

Learning objectives
  - Launch SageMaker Notebook
  - Install dependencies and import the libraries
  - Download the data and upload them to S3 bucket
  - Set up training and validation data
  - Configure and run the estimator
  - View the generated reports
                                                       Debugger
                                                       Hook
                                                         |          |---> Training
                     |----> train      --> S3 ---|       V          |     Report
  Shap --> Census  --|                           |---> SageMaker ---|
  library  Dataset   |----> validation --> S3 ---|       ^          |
                                                         |          |----> Profile
                                                        Rules              Report

   Shap Census Dataset
     - Standard UCI income dataset with 48K observations with 14 features
     - split 80% / 20% for trainining/testin

------------------
Solution
Launch SageMaker Notebook

    To avoid issues with the lab, open a new Incognito or Private browser window to log in to the lab. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region.
    In the search bar, type "SageMaker" to search for the SageMaker service. Click on the Amazon SageMaker result to go directly to the SageMaker service.
    Click on the Notebooks link under the Applications and IDEs section to view the notebook provided by the lab.
    Check if the notebook is marked as InService. If so, click the Open Jupyter link under Actions.
    Click on the eval.ipnyb file.

Install dependencies and import the libraries

Note: If you are prompted to select a kernel, please choose conda_tensorflow2_p310.

If this is your first time running a notebook, each cell contains Python commands you can run independently. A * inside the square braces indicates the code is running, and you will see a number once the execution is complete.

    Click the first cell that installs sagemaker, smdebug, numpy, and shap libraries. Use the Run button at the top to execute the code.

    The next cell imports the required Python libraries, initializes the sagemaker session and defines the output bucket. Click the cell and use the Run button to execute the code. This will import libraries from: Pandas, sklearn, and sagemaker.debugger. It may take a few minutes to complete the operation.

    The next cell fetches the IAM role using the get_execution_role function. Copy the following code snippet and paste into the empty code block and use the Run button to perform this operation.

    role = get_execution_role()

Download the data and upload them to the S3 bucket

    The next cell fetches the adult income dataset from the shap library and assigns them two variables X and y. Click Run to execute this cell.

    Now that the data is imported, it must be split for training and testing purposes. Copy the following code snippet and paste into the cell, and Run the cell to perform this operation.

    # Split into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    The next cell uses pd.concat to concatenate the train and test data along the first axis and assign them to variables train_data and validation_data, respectively. Click the cell and click Run to execute this cell.

    The next cell converts the training and validation data to CSV format. Click the cell and click Run to execute this cell.

    Next, we will upload the training CSV file to the S3 bucket using the upload_file function. Click the cell and click Run to upload the CSV to the S3 bucket.

    Copy the following code snippet and paste into the cell, and Run the code to upload the validation.csv file.

    s3.upload_file('validation.csv', output_bucket, f'{output_prefix}/validation/validation.csv')

Set Up Training and Validation Data

    Click the next cell and click Run to execute. This initializes the path from which training data and validation data will be read. These values will be used to create the input parameter for the estimator object.
    Then click the next cell and click Run to execute. This uses the TrainingInput function to create the train_data input parameter. Please pay attention; we are passing the path from which the training data will be read.
    Click the next cell and click Run to create an input parameter for validation_data.

Fetch the algorithm and initialize estimator

    Copy the following code snippet and paste above the print function. Click Run to execute. This will fetch the 1.2.1 version of xgboost algorithm.

    container = retrieve('xgboost', boto3.Session().region_name, version='1.2-1')

    The next cell sets the values for the hyperparameters and defines our objective metric. Select it and click `Run' to execute.
    Click the next cell and click Run to initialize the profiler_config, one of the estimator object's input parameters.
    Click the next cell and click Run to create the Estimator object and pass profiler_config, Debugger_hook_config, and rules.
    Finally, copy the following code snippet and paste into the next cell, and click `Run' to initiate the training process.

    xgboost_estimator.fit({"train": train_data, "validation": validation_data})

Note: This will take a few minutes to complete. Once complete, you can locate and download the xgboost_report.html and profiler-report.html files from the S3 bucket, to review and obtain further information.
------------------

  profiler-report.html and profiler-report.ipynb
  -> at:
      xgboost-iris-debugger-20250102-18-17-28-2025-01-02-18-17-46-238/ rule-output/ ProfilerReport/ profiler-output/

  xgboost_report.html and xgboost_report.ipynb
    -> at:
     xgboost-iris-debugger-20250102-18-17-28-2025-01-02-18-17-46-238/ rule-output/ CreateXgboostReport/


