------------------------------------------------------
3.2 Importing Data in the Data

  Cost: ~$1 USD

  Importing from S3
    - Create a new S3 bucket and upload data to S3

        -> AWS -> S3 -> Create Bucket -> Name: datawrangler-hotelbooking-ml-deep-pat -> create bucket
                     -> upload -> select demos/S03/start/hotel_bookings.csv -> upload

            - Create a new Data Wrangler Flow
               -> AWS -> SageMaker -> Domains -> Select [click in to] domain  ->  right click on "Launch" for selected user profile -> Studio
                 -> Home <left tab> -> Data -> Data Wrangler

            - Import the S3 Data into Data Wrangler

                 -> File -> New -> Data Wrangler Flow
                    Note: spun up ml.m5.4xlarge
                    -> rename Wrangler tab -> right-click -> rename -> New Name: HotelBookingFlow1.flow
                    -> Import data -> S3 -> bucket: datawrangler-hotelbooking-ml-deep-pat -> hotel_bookings.csv -> Import

          Importing from Athena
             - creating a database table in Athena using an AWS Glue crawler
                # create bucket to save Athen queries (if 1st time)
                -> AWS -> S3 -> Create Bucket -> Name: athena-queries-pat -> create bucket
                -> Athena -> Query editor -> Edit Setting -> Query location: s3://athena-queries-pat/hotel-booking-queries
                # create a new table using Glue Crawler
                -> Athena -> Query Editor ->  Create -> AWS Glue Crawler
                    <Glue Crawler> Name: HotelBookingS3Crawler -> Next -> Add a data source
                        -> Data Source: S3 , Bucket: datawrangler-hotelbooking-ml-deep-pat -> add data source -> Next ->
                         -> Create New IAM  Name: AWSGlueServiceRole-HotelBookings -> Create
                         # create a database called "default"
                         -> Target Database: default -> Next -> Create Crawler
                         -> Run Crawler
                         # when complete
                         -> Athena -> Query Editor  -> table: datawrangler_hotelboking_ml_deep_pat -> click on: <...> -> preview table

             - importing the Athena data into data wrangler
                -> Data Wrangler -> Import -> Athena -> Name: AthenaHotelBookings -> Connect
                   uncheck 'save connection' -> query: SELECT * FROM "default"."datawrangler_hotelbooking_ml_deep_pat" where arrival_date_year = 2015;
                   -> import -> dataset Name: AthenaBookings2015 -> add
                   -> Run

          Editing Data Types
             -> select data flow -> edit ->
                -> change "is_repeated_guest" from "long" to "boolean"
                -> preview (required)
          Joining Tables
            -> + -> join

  Deleting Resources
    - Deleting the Athena Data Source (table) and AWS Glue Crawler

    - Terminating instances and kernels for Data Wrangler

  Summary

     - A Data Wrangler flow can import data from a variety of sources
        -  used S3 and Athena but you can also pull from EMR, databrick, redshift, & snowflake
     - Once data is imported, it's possible to edit data types and join tables

------------------------------------------------------
3.3 Analyzing and Visualing Data in Data Wrangler

  Cost: ~$1 USD

  Data Quality and Insights Report
       -> AWS -> SageMaker -> Domains -> Select [click in to] domain  ->  right click on "Launch" for selected user profile -> Studio
         -> Home <left tab> -> Data -> Data Wrangler -> HotelBookingFlow1.flow
            -> Data Types -> click "+" -> Add Data Analysis -> Analysis Type: Data Quality and Insights Report, Target Column: is_cancelled,
               Problem Type: Classification, Name: HotelBookingIsCancelledInsightReport -> Create

               Duplicate Rows
               -> reports 28.1% of data is duplicated

               Target Leakage:
                  due to "is_cancelled" column '1' value always matches "reservation_status" column "Cancelled" value

               Anomalous Samples:
                  -> outlier data, example: adr (average daily rate) is 0

          -> File -> Save Data Wrangler Flow
          -> go back : at top click on "Data Flow"

  Histogram
    - Creating a Histogram of Data in Data Wrangler
        - cancelled by arrival date
        - cancelled by market segment

            -> Data Types -> click "+" -> Add Data Analysis -> Analysis Type: Histogram,
               X Axis: arrival_date_month, Colored by: is_cancelled, Name: HistorgramCancelledByArrivalMonth -> Run
               -> shows August has high cancellation

              -> File -> Save Data Wrangler Flow

               # 2nd histogram
               X Axis: market_segment, Colored by: is_cancelled, Name: HistorgramCancelledByMarketSegment -> Run
               -> shows: online_TA (travel agent) has largest cancellations, but
                         groop bookings have highest percentage cancelalation

               -> File -> Save Data Wrangler Flow

          -> go back : at top click on "Data Flow"

  Scatter Plot
     - Creating a Scatter Plot of Data in Data Wrangler
       - Total special requests and lead time
       - Booking changes and lead time

            -> Data Types -> click "+" -> Add Data Analysis -> Analysis Type: Scatter Plot,
               X Axis: total_of_special_requests, Y axis: lead_time, Colored by: is_cancelled, Name: ScatterPlotSpecialLeadTime -> Run

               -> File -> Save Data Wrangler Flow

            -> Data Types -> click "+" -> Add Data Analysis -> Analysis Type: Scatter Plot,
               X Axis: booking_changes, Y axis: lead_time, Colored by: is_cancelled, Name: ScatterPlotBookingLeadTime
                 Facet By: arrival_date_year -> Run

               -> File -> Save Data Wrangler Flow

  Delete Resources
    - terminate instances and kernels for Data Wrangler
      <running app icon>  -> shutdown apps
                             shutdown kernel

  Summary:

     Analysis and visualization makes it easier to understand our data
       - data Quality and Insights Report
       - Histograms
       - Scatter Plot

------------------------------------------------------
3.4 Transforming Data in Data Wrangler

   Cost ~$1 USD

   Drop Columns
     - create leakage analysis
     - create feature correlation analysis
     - determine columns to drop
     - Add a transform to drop the columns

         -> Home <left tab> -> Data -> Data Wrangler -> HotelBookingFlow1.flow
            -> Data Types -> click "+" -> Add Data Analysis -> Analysis Type: Target Leakage,
               Analysis Name: TargetLeakage,  Max Features: 32, Problem Type: Classification, Target: is_cancelled -> Run
               -> Possible redundant: total_of_special_requests, hotel, arrival_date_month, meal, and previous_cancellations

               -> File -> Save Data Wrangler Flow

            -> Data Types -> click "+" -> Add Data Analysis -> Analysis Type: Feature Correlation,
               Analysis Name: FeatureCorrelation,  Correlation Type: Linear -> Run
               Notes:
                    - You want features to be highly correlated with a target, is_canceled in our case, but not correlated
                      amongst themselves.
                    - anything above 0.9 is too highly correlated and should be removed including:
                       reservation_status, arrival_date_month, arrival_date_week_number, arrival_date_year, and reservation_status_date.


               -> File -> Save Data Wrangler Flow



      Columns to drop:
         Target leakage / possibly redundant (less than 0.5 correlation or predictive ability)
           - babies
           - hotel
           - arrival_date_day_of_month
           - meal
           - previous_cancellations
           - arrival_date_month
         Low Prediction Power
           - days_in_waiting_list
         Highly Correlated
           - arrival_date_week_number
           - arrival_date_year

            -> Data Types -> click "+" -> Add Transform -> Add Step -> Add Transform: Manage Columns ->
               -> Preview -> Add
               -> File -> Save Data Wrangler Flow

   Drop Duplicate Rows
     - using a managed transform to drop rows
     - Using a custom transform and pandas to view our new DataFrame
     - have 28.1% dumplicate rows

       # drop duplicate rows
       -> Add step -> Add Transform: Manage Rows -> transform: Drop Duplicates -> Preview -> Add

       # drop duplicate rows
       -> Add step -> Add Transform: Custom Transform -> Name: DataFrameInfo, Python(Pandas) -> code: df.info()  -> Preview
          -> shows: down to 15697 rows

   Handle Missing Values
     - Finding the missing values
     - determine what values to fill with
     - filling in the missing values

            # re-create insight report
            -> Data Types -> click "+" -> Add Data Analysis -> Analysis Type: Data Quality and Insights Report, Target Column: is_cancelled,
               Problem Type: Classification, Name: HotelBookingIsCancelledInsightReport -> Create

               -> in Feature summary: shows "agent" missing data 20.1%
               -> In "agent" table: shows 1 agent used in 63 % and prediction power was 0.433 (now just 0.152)

               -> <after previous transform> -> click "+" -> add transform  -> transform: Handle Missing -> add step
                  -> transform: Impute ,  Column Type: Numeric, Input Column: agent, Imputing Strategy: Approximate Median -> Preview -> Add
                    -> agent colunm should be filled in

          Note: Impute means:
              fill in the missing values with an estimated value based on our other data; so kind of taking a best guess


   Encode Categorical Data
     - some algorithms work better on encode values (1, 2, 3, ...) instead names
       example:
             Encoding   Market Segment    Market_Segment_Direct        Market_Segment_Corporate        Market_Segment_Aviation
               1            Direct               1                              0                              0
               2            Corporate            0                              1                              0
               3            Aviation             0                              0                              1
               4            Corporate            0                              1                              0


     - One-hot encoding categorical data
         - market_segment
         - assigned_room_type
         - deposit_type
         - customer_type
         - is_repeated_guest

          -> Add step -> Encode Categorical , transform: one-hot encoded,
            input columns: market_segment assigned_room_type deposit_type customer_type is_repeated_guest,
            output type: columns -> Preview -> Insert

            Note: is_repeat_guest caused an error, so I removed it

            -> rename new columns with spaces " " instead of "_"
            -> Add step -> manage columns ->  tranform: rename, input column: market_segment_Online TA,
               output Column: market_segment_Online_TA -> Preview -> add
               # repeat for market_segment_Online TA/TO, deposit_type_No Deposit, deposit_type_Non Refund

   Handle Outliers
     - finding features with outliers
     - handling the outliers

     Insight report shows: Children has 4.77% outliers
        -> Add Step -> Handle Outliers -> transform: Standard Deviation numeric outliers, input colunm: children,
           fix method: remove, Standard Deviation: 4 -> Preview -> Add
               -> File -> Save Data Wrangler Flow

   Delete Resources
    - terminate instances and kernels for Data Wrangler
      <running app icon>  -> shutdown apps
                             shutdown kernel

   Summary:

      Data Wrangler offers several types of transformst to clean up your data
        - drop columns
        - drop duplicate rows
        - handle missing values
        - encode categorical data
        - handle outliers

------------------------------------------------------
3.5 Exporting Data for Training

  Cost: $4 USD

  Four Options for Exporting
    Export to SageMaker Pipeline
      - for large-scale ML workflows
    Export to S3
      - for small datasets or processing jobs
    Export to Python Code
      - to manually integrate into a workflow
    Export to SageMaker Feature Store
      - for storing and sharing features centrally

   Export or Add Destination
       Export to:
          - Requires you to generate and run a Jupyter Notebook
          S3 (via Jupyter Notebook)
          SageMaker Pipelines (via Jupyter Notebook)
          Python Code
          SageMaker Feature Store (via Jupyter Notebook)
       Add destination:
         - after adding the destination, export the data flow in a few clicks (no Jupyter Notebook required)
          S3
          SageMaker Feature Store

  Exporting to SageMaker Pipeline (from Data Wrangler)
    - Viewing the Jupyter Notebook that gets generated

         -> Home <left tab> -> Data -> Data Wrangler -> HotelBookingFlow1.flow
             # must run data validation before exporting data
             -> "run validation"a <top right> -> Done

            -> <after last transform, click "+" -> Export -> SageMaker Pipelines (via Jupyter Notebook) -> Select (defaults)
              # creates a jupyter notebook
              -> not actually going to use Notebook, so close tab -> Save Notebook first


               -> File -> Save Data Wrangler Flow
  Export to S3
    - Using a destination node

            -> <after last transform, click "+" -> Add Destination -> Amazon S3 ->
              Destination Name: TransformedHotelData, bucket: s3://datawrangler-hotelbooking-ml-deep-pat/ -> Add destination
                 # create processing job
                 -> Create Job <top right> -> Next 2: Configure job -> Instance Type: ml.m5.4xlarge, Instance Count: 1 -> Create

                  # click on "process job name" link for job details
                  # wait for Status: Completed
                  -> check S3 bucket, will create CSV file in a new folder

    - Using export to
            -> <after last transform, click "+" -> Export to -> Amazon S3 (via Jupyter Notebook) -> Select (defaults)
              # creates a jupyter notebook
              -> Follow steps in Notebook, then process data will be placed in S3
              -> not actually going to use Notebook, so close tab

    - using export data
            -> <click last transform> -> Export Data <upper right> -> bucket: s3://datawrangler-hotelbooking-ml-deep-pat/
                -> export data
                # create a new folder with CSV data
                -> split to 4 CSV (?)

  Export to Python Code
    - viewing the generate code
    - useful when you have larger code base, and want to incorporate code

            -> <after last transform, click "+" -> Export to -> Python Code -> shows Python code with all the transforms

              -> right click -> select all -> copy ->

  Export to SageMaker Feature Store
    -> Viewing the Jupyter Notebook that gets generated

            -> <after last transform, click "+" -> Export to -> SageMaker Feature Store -> (defaults) -> Select

            Note: Create Feature Group
                alert: Record identifier and Exvent time feature name are required
                -> need these features to 1st be added to dataset

               -> File -> Save Data Wrangler Flow

   Delete Resources
    - terminate instances and kernels for Data Wrangler
      <running app icon>  -> shutdown apps
                             shutdown kernel
  Summary
    After transforming your data, Data Wrangler offers four options for exporting it
      - Export to SageMaker Pipeline
      - Export to S3
      - Export to Python Code
      - Export to SageMaker Feature Store

    Add destination:
         - after adding the destination, export the data flow in a few clicks (no Jupyter Notebook required)
         - use "add destination" to create a processing job instead of Jupyter Notebook
          Destinations:
           - S3
           - SageMaker Feature Store


