---------------------------------------------------
  Amazon Fraud Detector
---------------------------------------------------

AWS AI Services Programming Series - Part 5 (Fraud Detector)
AWS Tutorials - Detect Frauds using Amazon Fraud Detector AI Service
  https://www.youtube.com/watch?v=xsJ63CQmx_k

 AWS Dojo Workshops
   https://aws-dojo.com/workshoplists/
   -> contains tutorial and exercises using AWS Services including this tutorial

   Fraud detector workshop:
     https://aws-dojo.com/workshoplists/workshoplist17/


   Other Dojo AI Services tutorials:
   Part 3: Transcribe, Part 2: Lex, Part 1: Polly, Translate, & Textract,
   part 4: Comprehend, Part 6: Forecast, Part 7: Personalize, 

  Summary:
    In this workshop, you create a custom model using a training dataset stored in S3 bucket. 
    The model is deployed. You then build a client application to call the model using API to 
    detect the fraud.


  Amazon Fraud Detector:
    - a fully managed service that uses machine learning (ML) to identify potentially fraudulent 
      activity. 
    - automates the time consuming and expensive steps to build, train, and deploy an ML model 
      for fraud detection.
    - customizes each model it creates to a customer’s own dataset. 
    - The model can be called using API to detect the frauds in the applications.


  API call AWS AI services
     recognition:  video and image analysis
     polly: text to speech
     transcribe: speech to text
     translate: translate text from one language to another language
     fraud detector

     - for most of these services, you do NOT need to train  model, etc

  Fraud Detector has 3 parts:
  1. Build a Fraud Detector model
      - do need to train your model, but simplied by having a framework in place
      - training data is loaded to S3 bucket
      - need training data with a mix of events categorized as "legit" or "fraud"
      - then deploy this model and use API calls to this model
      - configuring fraud detector
          - select data to train model
          - identify features of the data to be used
          - select fraud detection algorithm
          - start training of your model
      - after training:
        - validate the performance [accuracy] of your model
        - if satisfied with performance, host your model

   2. Fraud Detector Detection Logic
     - Combine you model with decision rules to turn model scores into actionable outcomes

   3. Fraud Detector Prediction API 
     - call Fraud Detector API with online event data to receive fraud predictions
    

 Fraud Detction Model Types:
   - used to select machine learning type
    Online Fraud Insights
      - model type is optimized to detect fraud when little historical data is available about the entity being evaluated, 
        for example, a new customer registering online for a new account.

    Transaction Fraud Insights
      - model type is best suited for detecting fraud use cases where the entity that is being evaluated might have a 
        history of interactions that the model can analyze to improve prediction accuracy (for example, an existing 
        customer with history of past purchases).

    Account Takeover Insights
      - model type detects if an account was compromised by phishing or another type of attack. 
      - The login data of a compromised account, such as the browser and device used at login, is different from the 
        historical login data that’s associated with the account.


  Fraud Detector Dataset requirements
    - MUST have "EVENT_TIMESTAMP" (date and time when event occurred) 
    - MUST have "EVENT_LABEL" fields (represents whether event was legitimate or fraud)
    - MUST have at least 10K records

  Fraud Detector Event Type 
    - Before you create your fraud detection model, you must first create an event type. 
    - Creating an event type involves defining your business activity (event) to evaluate for fraud. 
    - Defining the event involves identifying the event variables in your dataset to include for fraud evaluation, 
      specifying the entity initiating event, and the labels that classify the event. 

  Fraud Detector Variables
    - Variables represent data elements that you want to use in a fraud prediction.
    - These variables can be taken from the event dataset that you prepared for training your model, from your 
      Amazon Fraud Detector model's risk score outputs, or from Amazon SageMaker models.
    - The variables you want to use in your fraud prediction must first be created and then added to the event 
      when creating your event type. 
    - Each variable you create must be assigned a datatype, a default value, and optionally a variable type. 
    
  Fraud Detector Entity
    - An entity represents a person or thing that's performing the event. 
    - An entity type classifies the entity. Example classifications include customer, merchant, user, or account. 
    - You provide the entity type (ENTITY_TYPE) and an entity identifier (ENTITY_ID) as part of your event dataset 
      to indicate the specific entity that performed the event.
  
    - Amazon Fraud Detector uses the entity type when generating fraud prediction for an event to indicate who 
      performed the event. 
    - The entity type you want to use in your fraud predictions must first be created in Amazon Fraud Detector and 
      then added to the event when creating your event type.

   Fraud Detector Model Validation
     - by default,  it validates model performance using 15% of the data that was not used to train the model. 
     - When you try to detect an event for fraud using this model, it returns a score distribution and a confusion matrix

   Fraud Detector  Model performance metrics
     Score distribution chart 
       – A histogram of model score distributions assumes an example population of 100,000 events. 
       - The left Y axis represents the legitimate events and the right Y axis represents the fraud events. 
       - You can select a specific model threshold by clicking on the chart area.
     Confusion matrix 
       – Summarizes the model accuracy for a given score threshold by comparing model predictions versus actual results
     Receiver Operator Curve (ROC) 
        – Plots the true positive rate as a function of false positive rate over all possible model score thresholds. 
        - View this chart by choosing Advanced Metrics.
     Area under the curve (AUC) 
       – Summarizes TPR and FPR across all possible model score thresholds. 
       - A model with no predictive power has an AUC of 0.5, whereas a perfect model has a score of 1.0.
     Uncertainty range 
       – It shows the range of AUC expected from the model. 
       - Larger range (difference in upper and lower bound of AUC > 0.1) means higher model uncertainty. 
       - If the uncertainty range is large (>0.1), consider providing more labeled events and retrain the model.
     
  Building Fraud detector in workshop
     https://aws-dojo.com/workshoplists/workshoplist17/

      S3               --->  Fraud Detector     ---> Jupyter Notebock
      sample training        Model + Detector        Python Client
      Data - from AWS                                 to call detector logic



  AWS S3 -> Create bucket -> pat-demo-bkt -> Create bucket
    -> update "sample-data.csv"
       # note: dataset has 20k records with 18996 legit 

1. Create Fraud Detector Event Type (Create Event Type [add Entity , add data location, specify variables, add labels])

  AWS -> Fraud Detector -> Events -> Create Event Type ->
      Name: dojo-event
      Entity -> Create New enity -> Entity Type Name: dojoentity -> Create Entity
      Entity: dojoentity,
      Event Variables: Choose how to define this event's variables: Select variables from a training dataset
      IAM Role -> Create IAM Role: S3 bucket (to access): pat-demo-bkt -> Create Role
        Role Created: AmazonFraudDetector-DataAccessRole-1727212818834
           -> allows access to S3 arn:aws:s3:::pat-demo-bkt bucket for ListBucket, GetBucketLocation, & GetObject
      Data Location:
        s3://pat-demo-bkt/sample-data.csv

       # variables are the remaining columns after EVENT_TIMESTAMP and EVENT_LABEL
       #   can remove potential variables for dataset columns that would not impact fraud
       Variable(2)	Variable type
       ip_address       IP Address
       email_address    Email Address
	
       # minimum of 2 labels (for fraud and legitimate, but can have more ??)
       Labels
         Create Label -> legit -> create
         Create Label -> fraud -> create
     
        -> Create Event Type


2. Build [train] Fraud Detector model (assign Model Type, assign Event Type, training data location, 
   configure: [Model inputs & label classification] & build

        -> Build a Model
           -> Model Name: dojo_fraud_detection_model, Model type: Online Fraud Insights, 
             Event Type: dojo-event
             IAM Role: AmazonFraudDetector-DataAccessRole-1727212818834
             Training data location: s3://pat-demo-bkt/sample-data.csv
             -> Next ->
             Configure Training: 
                Model Inputs:
                   Variable	Variable type
	            ip_address	IP Address
	            email_address	Email Address
              Label Classification
                Fraud label: fraud, Legitmate label: legit
                -> Next -> Review and Create 
                    -> Create and train model 
                    # takes about 50 minutes

3. Examine Model performance
  AWS -> Fraud Detector -> model -> dojo_fraud_detection_model -> version 1.0 -> Model Performance ->
       -> Examine Score distribution graph
           - in demo: you will suceed in catching 72.7 of all fraudulent events while accepting a risk that 13.5 of
             legitimate events

3. Deploy Model

  AWS -> Fraud Detector -> model -> dojo_fraud_detection_model -> version 1.0 -> Actions -> Deploy Model Version
    -> Deploy Version

4. Create Fraud Detector

  AWS -> Fraud Detector -> Detectors <left tab> -> Create Detectors ->
     Detector Name: dojodector, Event Type: dojo-event -> Next ->
     Add Model -> dojo_fraud_ detection_model, version: 1.0 -> Add model -> Next ->
     Add rules: 
        Name: highriskrule, Expression: $dojo_fraud_detection_model_insightscore > 800
     Outcomes: Create a new outcome
        Outcome name: high_risk -> Save outcome
     -> Add rule

     Add another rule: 
        Name: mediumriskrule, Expression: $dojo_fraud_detection_model_insightscore <= 800 and $dojo_fraud_detection_model_insightscore > 500
     Outcomes: Create a new outcome
        Outcome name: medium_risk -> Save outcome
     -> Add rule

     Add another rule: 
        Name: lowriskrule, Expression: $dojo_fraud_detection_model_insightscore <= 500
     Outcomes: Create a new outcome
        Outcome name: low_risk -> Save outcome
     -> Add rule
     
     -> Next -> 
       Configure rule execution: 
         Rule execution models: First matched, keep rule order (high, medium, low)
     -> Next -> 
        Create Detector 
        # publish detector
        dojodetector (version 1) -> Action -> Publish   
        # after publishing, 'dojodector verion 1.0' detector status should now be Active
            
     # test detector
       # test 1:
          timestamp: 2024/09/24 02:00:00 (yesterday)
          entityId:  1234
          email_address: dummy@myemail.com
          ip_address: 127.0.0.1
            -> outcome: low_risk Model score: 224
   
       # test 2:
          timestamp: 2024/09/24 04:00:00 (yesterday)
          entityId:  2929
          email_address: fred@example.com
          ip_address: 36.19.221.248
            -> outcome: medium_risk Model score: 670
   
4. Create Client

   AWS -> IAM -> Roles -> Create Role -> 
      Trusted Entity: AWS Service, Service or Use Case: SageMaker , Use Case: SageMaker - Execution
      # this attaches AmazonSageMakerFullAccess policy 
      -> Next -> Permissions -> Next -> Tags -> Review: Role Name: dojosagemakerrole -> Create Role

      -> Open the dojosagemakerrole role details,  -> Add Permission -L 
        remove AmazonSageMakerFullAccess policy and attach PowerUserAccess policy to the role.


   AWS -> SageMaker -> Applications and IDE <left tab> -> Notebooks  -> Create Notebook instance ->
      Notebook Instance Name: dojofraudclientnotebook , instance type: ml.t3.medium
      IAM role: dojosagemakerrole, Root access: Enable - give users root access to the notebook -> create Notebook instance

      dojofraudclientnotebook  -> Open Jupyter -> New <upper left> -> condo_python3

      In the code below, you create client for frauddetector and use get_event_prediction method to find risk for the 
        email_address and ip_address passed as the parameters. You are also passing detector name, entity type and event 
        type as the parameters. You then print the response of the get_event_prediction method call which will have the score 
        and outcome for the risk level of the event (email_address and ip_address).

Code: fraud detector jupyter notebook code

import boto3
import uuid

ENTITY_TYPE    = "dojoentity"
EVENT_TYPE     = "dojo-event" 

DETECTOR_NAME = "dojodetector"
DETECTOR_VERSION  = "1"

eventId = uuid.uuid1()

fraudDetector = boto3.client('frauddetector')

response = fraudDetector.get_event_prediction(
detectorId = DETECTOR_NAME,
eventId = str(eventId),
eventTypeName = EVENT_TYPE,
eventTimestamp = '2020-07-13T23:18:21Z',
entities = [{'entityType':ENTITY_TYPE, 'entityId':str(eventId.int)}],
eventVariables = { 'email_address' : 'johndoe@exampledomain.com', 'ip_address' : '192.10.10.24'})

print(response)

# response:

{'modelScores': [{'modelVersion': {'modelId': 'dojo_fraud_detection_model', 'modelType': 'ONLINE_FRAUD_INSIGHTS', 'modelVersionNumber': '1.0'}, 'scores': {'dojo_fraud_detection_model_insightscore': 160.0}}], 'ruleResults': [{'ruleId': 'lowriskrule', 'outcomes': ['low_risk']}], 'externalModelOutputs': [], 'ResponseMetadata': {'RequestId': '6525552c-3c06-4617-8ce5-2a4a864ed0ae', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Wed, 25 Sep 2024 19:01:13 GMT', 'content-type': 'application/x-amz-json-1.1', 'content-length': '288', 'connection': 'keep-alive', 'x-amzn-requestid': '6525552c-3c06-4617-8ce5-2a4a864ed0ae'}, 'RetryAttempts': 0}}


5. Clean up

    Delete the dojofraudclientnotebook notebook instance.
    Delete the dojosagemakerrole IAM role.
    Delete the dojo-fraud-records S3 bucket.
    Delete dojodetector, dojo_fraud_detection_model, dojo-event, dojoentity in the Amazon Fraud Detector console.
      -> Must deactive dojodetector v1 First
      -> Next delete dojodetector (Version 1) 
      -> Next: delete Associated Rules: [high,medium,low]riskrule (Version 1) 
      -> Next: delete dojodector
      -> Next: Undeploy dojo_fraud_detection_model(version 1.0)
      -> Next: delete dojo_fraud_detection_model(version 1.0)
   


---------------------------------------------------


