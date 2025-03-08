------------------------------------------------------
4.3 SageMaker's Algorithms in Action

  XGBoost Algorithm
  https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html

  The machine Learning lifecycle
    Algorithm vs Model
      -> Training Data    ->  Algorithm                   ->    Model
                             - Code that identifies           - The Output from running the algorithm on the data
                               patterns in the data           - Rules, numbers, and data structures required to make predictions

  Creating a Training Job in SageMaker Studio

    Steps to Create a SageMaker Training Job
      1. Specify S3 bucket URLs for:
          S3 bucket containing training data
          S3 bucket to store model output/artifacts created during training process
      2. Specify the algorithm and compute instances to use for the training
      3. Specify the path to the [training] code stored in the Elastic Container Registry (ECR)

  XGBoost Algorithm:
    https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html
      For EC2 type recommendations, see "EC2 Instance Recommendation for the XGBoost Algorithm" section
      For Docker Registry Paths, see "Supported Versions" section which contains, the following link:
       "Docker Registry Paths and Example Code":
          https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/sagemaker-algo-docker-registry-paths.html
            -> Topics [regions] -> "US East (N. Virginia)  -> Topics [algorithms] -> XGBoost -> ECR Paths

  Evaluating the Model
    Goal:
       - A model that generalizes well so it makes accurate predictions
    Sometimes models:
        Overfitting
          - too tightly predicts
          - too dependent on training data and not likely to perform well on new data
          To prevent overfitting:
            - use more [training] data
            - add some "noise" to data [data augmentation to make dataset larger]
            - remove features
            - Early stopping [before it overfits the data]

        Underfitting
          - the model is too simple and doesn't accurately reflect the data [has not learned enough]
          To prevent underfitting:
            - Use more data
            - add more features
            - train longer


    Tuning the Model

         Training  -------------> Evaluation ---------------> Prediction
             |                        |
             |            Do we need to change the algorithm?
             |            Do we need to do more feature engineering?
             |            Do we need new or different data?
             |                        |
             |------------------------|

       Hyperparameters
           - the "knobs" you can use to control the behavior of your algorithm
           - XGBoost Hyperparameters:
             https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html

             XGBoost Automatic model tunning:
             https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
               - Automatic model tuning, also known as hyperparameter tuning, finds the best version of a model by running many
                 jobs that test a range of hyperparameters on your training and validation datasets.
               - You choose three types of hyperparameters:
                  - a learning objective function to optimize during model training
                  - an eval_metric to use to evaluate model performance during validation
                  - a set of hyperparameters and a range of values for each to use when tuning the model automatically

      Hyperparameter Tuning in SageMaker
           1. Choose your object metric
             - the thing you are trying to optimize
           2. Define ranges for hyperparameters
           3. SageMaker runs training jobs to optimize for the objective metric
             - till it finds the right combination of values to optimize your object metric

      How did the model do with Predictions?

        Confustion Matrix:

                        |  Predicted Class     |     Predicted Class
                        |  Positive            |     Negative
                        |  (FRAUD)             |     (NOT FRAUD)
          --------------|----------------------|----------------------
          Actual Class  | True Positive (TP)   |  False Negative (FN)
                        |                      |
          Postive       | FRAUD was correctly  | FRAUD was incorrectly
          (FRAUD)       | predicted as FRAUD   | predicted as NOT FRAUD
                        |                      |
          --------------|----------------------|----------------------
          Actual Class  | False Positive (FP)  |  True Negative (TN)
                        |                      |
          Negative      | NOT FRAUD was        | NOT FRAUD was correctly
          (NOT FRAUD)   | incorrectly predicted| predicted as NOT FRAUD
                        | as FRAUD             |
          --------------|----------------------|----------------------

        Metric for Classification Problems

           Accuracy: (TP + TN)  / (TP + FP + TN + FN)
              - percentage of predictions that were correct:
              - less effective with a lot of true negatives
                 - example: predicting fraud with little to no fraud data

           Precision: (TP)  / (TP + FP)
              - percentage of positive predictions that were correct:
              - Use when the cost of false positives is high
                 - example: an email is flagged and deleted as spam when it really isn't

           Recall: (TP)  / (TP + FN)
              - percentage of actual positive predictions that were correctly identified:
              - Use when the cost of false negatives is high
                 - example: someone has cancer, but screening does not find it

  Training and Evaluationg the Model for Customer Churn using XGBoost
    cost: ~$1 USD
    Training and evaluating the model in SageMaker Studio
      - Create a training job:
        - 1st, through the UI
        - 2nd, through a SageMaker Notebook
      - Train the model
      - Evaluate the model

      # walking through using the UI:
      -> SageMaker -> Training <left tab> -> Training Jobs -> Create training Job ->
        Job Name:   XGBoostCustomerChurn, IAM Role: <default>, Algorithm Source: <default>,
        Algorithm: Tabular - XGBoost:v1.3, Container: <automatically filled based on region>,
        Input Mode: File, Instance Type: ml.m4.xlarge, Instance Count: 1,
        Hyperparameters: objective (must be set): binary:logistic (meaning: binary: working with binary problem
           - churn/not churn; logistic: return a probability)
           Input data configuration: Data Source: S3, bucket: TBD,
           Output data configuration: S3 bucket: TBD
           -> not completed, but being showned

      Note: Input Mode:
         File: Training data will be download before it starts
         Pipe: Streams your data to the algorithm while it is running; gives faster performance

      # Setup S3 Bucket For Jupyter Notebook
         S3 -> Create Bucket -> bucket name: xgboost-customer-churn-ml-deep-pat -> Create Bucket

      # using the SageMaker Studio Jupyter Notebook:
        -> AWS -> SageMaker -> Domains -> Select [click in to] domain
        ->  right click on "Launch" for selected user profile -> Studio
                 -> Home <left tab> -> Folder -> S04/xgboost/start/xgboost_churn.ipynb <double click>
                  <defaults> -> select
         # in Jupyter Notebook:
           Initialize Environment and Variables:
             set bucket name: bucket = "xgboost-customer-churn-ml-deep-pat"
             # note data has already been split train and validation CSV files

Train
-> set location of training data

# The location of our training and validation data in S3
s3_input_train = TrainingInput(
    s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv'
)
s3_input_validation = TrainingInput(
    s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv'
)


-> set location of XGBoost container

# The location of the XGBoost container version 1.5-1 (an AWS-managed container)
container = sagemaker.image_uris.retrieve('xgboost', sess.boto_region_name, '1.5-1')

             -> Initialize Hyperparameters, setup output bucket, and set up Estimator
# Initialize hyperparameters
hyperparameters = {
                    'max_depth':'5',
                    'eta':'0.2',
                    'gamma':'4',
                    'min_child_weight':'6',
                    'subsample':'0.8',
                    'objective':'binary:logistic',
                    'eval_metric':'error',
                    'num_round':'100'}

# Output path where the trained model will be saved
output_path = 's3://{}/{}/output'.format(bucket, prefix)

# Set up the Estimator, which is training job
xgb = sagemaker.estimator.Estimator(image_uri=container,
                                    hyperparameters=hyperparameters,
                                    role=role,
                                    instance_count=1,
                                    instance_type='ml.m4.xlarge',
                                    output_path=output_path,
                                    sagemaker_session=sess)
-> Run training:

# "fit" executes the training job
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})

-> During training it reports

[1]#011train-error:0.05058#011validation-error:0.08108
[2]#011train-error:0.04886#011validation-error:0.07508
.  .  .
[98]#011train-error:0.02829#011validation-error:0.06607
[99]#011train-error:0.02829#011validation-error:0.06607

-> when finished it reports 0.028 error rate on training data (0.982 accuracy)
->               it reports 0.066 error rate on validation data (0.944 accuracy)

# train model
-> Go to S3 bucket  demo/output/sagemaker-xgboost-<date time>/ model.tar.gz

  Delete Resources
     - Terminating instances and kernels

  Summary:
    Training a model in SageMaker starts with a training job
      - created through the SageMaker UI or  a SageMaker Notebook
      - specify the dataset, the algorithm, compute instance, ane ECR path
    Evaluate the model to see how well it makes predictions
      - use hyperparameters tunning to get it just right


