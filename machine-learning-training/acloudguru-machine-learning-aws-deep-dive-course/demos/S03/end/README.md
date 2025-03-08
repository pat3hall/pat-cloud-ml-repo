------------------------------------------------------
3.7 Ingesting and Accessing Features in SageMaker Feature Store

  Cost: ~$1 USD

  Generate Features required by Feature Store

    - In Data Wrangler, add custom transform to insert features for record_id and event_time
    - Export Flow to SageMaker Feature Store
    - View the Jupyter Notebook that was generated


       -> AWS -> SageMaker -> Domains -> Select [click in to] domain
         ->  right click on "Launch" for selected user profile -> Studio
         -> Home <left tab> -> Data -> Data Wrangler -> HotelBookingFlow1.flow ->

            -> Run validation
            -> <after last transform, click "+" -> Export to -> SageMaker Feature Store -> (defaults) -> Select

            Note: Create Feature Group
                Alert: Record identifier and Event time feature name are required
                -> need these features to 1st be added to dataset, both are set to "None"

                record_identifier_feature_name = None
                  -  the record identifier is just the unique ID for each record,
                event_time_feature_name = None
                  - event_time is a timestamp that allows you to perform time travel basically and versioning of the feature


            -> <after last transform, click "+" -> Add Transform -> Add step -> custom transform ->
               # create a new record_id column incrementing id by 1 for each row
               # code is at:  demos/S03/end/InsertRecordID.py
               Name: InsertRecordID
               Optional: Python (Pandas)
               code: df.insert(0,'record_id',range(1, 1 + len(df)))

               # create a new event_time column with time set to now
            -> Add step -> custom transform ->
              Name: InsertEventTime
              Optional: Python (Pandas)
                      code:
++++++++++++
import pandas as pd
import datetime

timestamp = pd.to_datetime("now").timestamp()

df.insert(1,'event_time', timestamp)
++++++++++++

        # create a new Jupyter Notebook
        -> close tab for previous Jupyter Notebook

        -> back to "data flow"
        -> <after last transform, click "+" -> Export to -> SageMaker Feature Store -> (defaults) -> Select


    - Update the record_id and event_time features
    - create a feature group through the SageMaker UI
    - Run the code in the Jupyter Notebook to create feature group

     # under Feature group's code, set record id and event time feature names:
        record_identifier_feature_name = "record_id"

        event_time_feature_name = "event_time"

     -> run blocks of code in "define feature group", "feature definitions",

  Create a Feature Group

     -> Configure Feature group from UI
        -> Home <tab> -> Data -> Feature Store ->  Create Feature Group -> Feature Group Name: my-demo-feature-group
           Storage Configuration: Online storage (other defaults) -> Continue ->
           Table: feature name     Feature type
                  customer_name     string
                  age               integral
                  event_time        string
                  -> Continue
                  Required Features:   Record Identifier Feature Name: customer_name, Event Time Feature Name: event_time
                  -> Continue -> Create Feature Group
                  -> View Feature Group

       # Note: Feature Group would normally be created via code (not using UI)
     -> Configure Feature group from Jupyter Notebook
       -> back to Jupyter Notebook <tab> under "Configure Feature Group"
           -> run code to create feature group
             -> created: Feature Group Name: FG-HotelBookingFlow1-9bc4ee8e
             -> run code blocks to initialize, etc
             -> wait for "FeatureGroup FG-HotelBookingFlow1-d2ec5f84 successfully created."
             -> Jupyter Notebook create feature group includes all features (columns)

  Ingesting into Feature Store
       -> back to Jupyter Notebook <tab> under "Inputs and Outputs"
         -> run code blocks
         # next: uploading the flow to S3 so the processing job can grab it
         -> run code in "Upload Flow to S3"
         # configure and run processing jobs
          -> in Job Configuration code block: change instance count from 2 to 1
          instance_count = 1
          -> run Configure blocks
          -> skip optional Spark Cluster Driver Memory block
          # run processing job blocks and then "job Status & S3 Output Location" block
           -> wait for complete

   # wait for processing job to complete
   SageMaker-> Processing <left tab> -> Processing Jobs -> data-wrangler-flow-processing-17-22-03-17-bed76aa2

  Accessing Features from Feature Store

  -> add end of HotelBookingFlow Export to Feature Store Notebook add:

     Code: Dump record_id 7 contents  from online  Feature Store:

       record_id = '7'
       featurestore_runtime =  sess.boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)

       record_id = featurestore_runtime.get_record(
           FeatureGroupName=feature_group_name, RecordIdentifierValueAsString=record_id)

       record_id['Record']

       # dump record_id 7:

       [{'FeatureName': 'record_id', 'ValueAsString': '7'},
        {'FeatureName': 'event_time', 'ValueAsString': '1721771486.699969'},
        {'FeatureName': 'is_canceled', 'ValueAsString': '0'},
        {'FeatureName': 'lead_time', 'ValueAsString': '266'},
        {'FeatureName': 'stays_in_weekend_nights', 'ValueAsString': '1'},
        .  .   .
        {'FeatureName': 'customer_type_Group', 'ValueAsString': '0.0'},
        {'FeatureName': 'is_repeated_guest_0', 'ValueAsString': '1.0'},
        {'FeatureName': 'is_repeated_guest_1', 'ValueAsString': '0.0'}]




  Delete Resources
     - delete files under folder icon
     - Terminating instances and kernels
     - deleting the Athena Table
       -> Athena -> select table -> delete table
     - Deleting extra files

   -> AWS Console -> Athena  -> Editor -> Data Source: AwsDataCatalog, Database: sagemaker_featurestore,
           table: fg_hotelbookingflow_9a1aec90_1721771143 -> right click on dots -> Preview Table

       -> This runs the following Query:
          SELECT * FROM "sagemaker_featurestore"."fg_hotelbookingflow_9a1aec90_1721771143" limit 10;


  Summary

     SageMaker Feature Store
       - Requires features for record_id and event_time
         - created using a custom transform with pandas code

   Feature groups can be created through the SageMaker UI or a Jupyter Notebook

   A SageMaker processing Job will save the flow to S3 and the transformed data to Feature store

