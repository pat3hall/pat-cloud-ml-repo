2.3 Setting Up SageMaker Canvas

  Overviewing the Dataset
    Customer Churn scenario
      Cell Phone Company
        - Wants to predict whether customers will leave
        - has data on customer plans, usage, and support calls

    The Basic Machine Learning Flow
      Training data:
        - phone plan, day minutes, evening minutes, International charges
      Algorithm identifies patterns
      Trained Model - predicts if customer is going to leave

    Demo Dataset: demos/S02/start/churn-All.csv (~3300 rows)
         features (columns): State,Account Length,Area Code,Phone,Int'l Plan,VMail Plan,VMail Message,Day Mins,
                             Day Calls,Day Charge,Eve Mins,Eve Calls,Eve Charge,Night Mins,Night Calls,Night Charge,
                             Intl Mins,Intl Calls,Intl Charge,CustServ Calls,Churn?

  Launching the SageMaker Canvas App
     - SageMaker Canvas provides a 2-month free tier
     - Ouside the free tier, the ~cost for this lesson $0.50


    To launch:
      -> AWS -> SageMaker -> Canvas -> Select user profile -> Open Canvas
      OR
      -> AWS -> SageMaker -> Domains -> Select [click in to] domain  ->  right click on "Launch" for selected user profile -> Canvas

    to logout
       -> click on "logout" (top right)


------------------------------------------------------
2.4 Running Predictions in SageMaker Canvas

  SageMaker Canvas Pricing Page
    https://aws.amazon.com/sagemaker/canvas/pricing/
  cost - ~$30

  session charge: 1.90 /hr

  Training charge
      Number of cells	Price
    First 10M cells	$30 per million cells
    Next 90M cells	$15 per million cells
    Over 100M cells	$7 per million cells

  Splitting the Dataset and Upload It to S3
     - split data to training (70%) [churn-TRAINING.csv] and test (30%) [churn-TESTING.csv] sets

  Create the Model in Canvas [for this dataset]
  -> AWS -> S3 -> Create Bucket -> Name: sagemaker-canvas-ml-deep-pat -> create bucket

    -> select "sagemaker-canvas-ml-deep-pat" bucket -> upload -> select "churn-TRAINING.csv" & "churn-TESTING.csv" -> upload

  Performing a Quick Build [and output a model]
    - Creating a new model in Canvas
    - Importing the dataset from S3

  Running Predictions

  Deleting Resources and Logging Out of Canvas

    Launch Canvas:
      -> AWS -> SageMaker -> Domains -> Select [click in to] domain  ->  right click on "Launch" for selected user profile -> Canvas

      -> Models <left tab> -> New Model -> Customer-Churn -> Create Dataset -> Tabular -> name: churn-TRAINING -> Import -> S3 -> "sagemaker-canvas-ml-deep-pat" -> "churn-TRAINING.csv" -> create dataset
         -> select dataset

  Note: Now you need to upload dataset before creating model so you can assign dataset to model when creating the model!!!

   -> select "Customer-Churn" model ->
       select predict column: Churn?

   Performing a build
     - Standard build takes 2 - 4 hours
     - Quick build takes 2 - 15 minutes, but models can't be shared
        -> <top-right> Quick Build -> Quick Build

   Running Predictions
     - Importing the test dataset
     - reviewing the prediction results
       -> Predict -> Select -> Manual -> select "churn-TESTING" -> Generate Predictions -> View

  Delete Resources and logout of Canvas
    -> deleting the model and datasets in canvas
    -> logout of Canvas
    -> Empty bucket and delete bucker

  Summary
    Preparing the Data
      - split the churn dataset 70/30 into training and test sets
      - created an S3 bucket and uploaded the CSV files
   Creating the Model
     - imported th training data from S3
     - Used quick build to create the model
   Running Predictions
     - Used the test data to get predictions on whether customers will churn
   Deleting Resources and Logging Out


