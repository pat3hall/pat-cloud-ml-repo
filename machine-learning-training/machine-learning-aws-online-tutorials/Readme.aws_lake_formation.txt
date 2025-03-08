-------------------------------------
AWS Lake formation info
-------------------------------------

###################################################
Deep Dive Into AWS Lake Formation - Level 300 (28 min)
  Syed Jaffry - Solution Arch - AWS
  https://www.youtube.com/watch?v=Aj5T5fcZZr0

  Summary:
    - In this session, learn how to build a secure and automated data lake using AWS Lake Formation.
    - Also learn how to set up periodic sales data and ingest into the data lake, build automated transformations,
      and generate sales forecasts from the transformed data using AI


  Why AWS Lake Formation

     Example Use Case: Build a modern sales forecasting capability

        Data source
          - Relational Database with inventory and history data
        transform
          - transform raw data so it can be used for forecasting
        AI service
          - may use AI services to deliver a better outcome (e.g. Amazon Forecast)


    Typical steps for building a data lake
      1. Setup [S3] storage
        - setup your data lake [setting up your S3 buckets]
      2. Move data
        - ingest your raw source database into your data lake
        - build an ingest logic that can be run repeatedly robostly and more resiliently
      3. cleanse, prep, and catalog data
        - cleanse, prep, and transform the data so it can be used by Amazon forecast
      4. configure and enforce security and compliance policies
         - implement who has access to what data and what level of access do they have
         - how to manage that access going forward that easy and scalable
      5. Make data available for analytics
         - make data available for collaboration so they can discover each other's data


    AWS Lake Formation
      Overview
        - data lake lifecycle stages and activities
        Stage 1
          Ingest and Register data
        Stage 2
          Security and control
        Stage 3
          Collaborate and use
        Stage 4
          Monitor and audit

     Summit Demo:
        Created 2 buckets:
          summit-demo-landing
            - ingest raw data into and it's going to land
            - contains folder for each department including "retail"
          summit-demo-processed
            - store transformed data
          summit-demo-published
            - store results of Amazon forecast
            - end users will draw reporting and visualization from this data

     AWS Console -> AWS Lake Formation -> Data Lake setup
        -> Stage 1 Register your S3 Storage ->
             # register landing location
             Register Location -> Amazon S3 path:  s3://summit-demo-landing/retail -> Register Location
             # register processed location
             Register Location -> Amazon S3 path:  s3://summit-demo-processed       -> Register Location
             # register publish location
             Register Location -> Amazon S3 path:  s3://summit-demo-publish        -> Register Location
        -> Stage 2 Create a Database
             # create landing database
                Create a Database -> Name: summit-demo-landing-db, Location: s3://summit-demo-landing/retail,
                  description: landing database  -> Create database
             # create processed database
                Create a Database -> Name: summit-demo-processed-db, Location: s3://summit-demo-processed,
                  description: processed database  -> Create database
             # also create published database
                Create a Database -> Name: summit-demo-published-db, Location: s3://summit-demo-published,
                  description: published database  -> Create database
        -> Stage 3 Grant Permissions ->
            # grant access to the databases
            -> Grant permission ->
              # grant permissions to landing database
              Iam Users and Roles: <AWS_LF_WORKFLoW_ROLE>    Database: summit-demo-landing-db
                # 'super' combines 'create table', 'alter', and 'drop'
                Database Permissions: Super, Grantable Permissions: <None - default> -> Grant
              # grant permissions to processed database
              Iam Users and Roles: <AWS_LF_WORKFLoW_ROLE>    Database: summit-demo-processed-db
                # 'super' combines 'create table', 'alter', and 'drop'
                Database Permissions: Super, Grantable Permissions: <None - default> -> Grant
              # grant permissions to published database
              Iam Users and Roles: <AWS_LF_WORKFLoW_ROLE>    Database: summit-demo-published-db
                # 'super' combines 'create table', 'alter', and 'drop'
                Database Permissions: Super, Grantable Permissions: <None - default> -> Grant
        # ingest data
          -> Ingestion <left tab bottom> -> Blueprints -> Use blueprint
          # blueprint provides option to connect to a relational database
          # incremental: Load new data to your data lake from MySQL, PostgreSQL, Oracle, and SQL Server databases.
             -> Incremental database
               Database connection: sales-db-conn # previously created Glue connection
               source datapath: <SQL database>/<table>
               increment data: <incremental bookmark info>
               target database: summit-demo-landing-db/retail, format: Parquet
               import frequency: Daily
               workflow name: demo-wf1, IAM Role: AWS_LF_WORKFLOW_ROLE, table prefix: demo -> Create
               # blueprint creates an orchestrated flow to ingest the source database to your data lake DB
               -> Start Now
               # examine workflow
                 click on "demo-wf1" -> click on its definition  -> Run View

        # Create Glue workflow
        AWS Console -> Glue -> ETL <left tab> -> Workflows -> Add workflow ->
            # demo had pre-created the workflow

            -> Run workflow



      Inventory forecast end to end solution
     -----------------------------------------------------------------------------
     | Lake Formation          AWS Lake Formation                                |
     |    Blue Print                                                             |
     |  ^  \         ----------- Lake Formation Security ---------------------   |
     --/----\-------/-----------------|---------------------------------------\--|
      /  1   |     /                  |                                        \
     /       V    /                   |              Amazon Forecast            \
    ERP      S3          Glue Job    S3           Load  Train  Generate          S3
             Landing -> transform -> Processed -> data  Model  Forecast -->  Published
             Bucket      data        Bucket        ^       ^      ^      5    Bucket
               |          ^                        |       |      |
               | 2        | 3                      ----------------
               V          |                                | 4
     -----------------------------------------------------------------------------
     |                                                                           |
     |                       AWS Glue Workflow                                   |
     |                                                                           |
     -----------------------------------------------------------------------------

     Lake / Glue Workflow Steps:
       1. Blue print ingests raw [ERP] data to Landing bucket on regular basis
       2. Kickoff Glue workflow
       Note: AWS Glue workflow orchestrates the rest of the flow
       3. Transform landing data to format useable by Forecast and storing in Proceed bucket
       4. Run Forecast flow
       5. Publish Forecast results to Publish bucket


      Note: ERP: Enterpise Resource Planning

     AWS Glue workflow for inventory forecast

                                        Amazon Forecast
             1               2   <------------------------------>    3
       S3    -->  AWS Glue  ---> Import  -->  Train  --> Generate   --->  AWS Glue Crawler
    new raw       Transform      dataset      predictor   forecast           Crawl
    data arrives    data                                                  exported forecast
       |             ^
       |triggers     |  Starts AWS Glue workflow
       V             |
     AWS Lambda -----|
     Start Workflow


     Glue Workflow Steps:
       1. transform landing raw data to format useable by Amazon Forecast
       2. Kickoff Amazon Forecasts 3 steps (import, train predictor, generate forecast/export results
          to publish bucket)
       3. Kickoff Glue Crawler
           Crawl: runs on the exported forecast dataset so that it is useable by SQL and Athena
                   and visualized by QuickSight


  Ingest and transform

    Ingest source relational database data:
      1. run a select * statement on your source database and bring the data in
        - approached used by blueprint
        - with this approach on a very busy database, you may end up missing the data from a live transaction
        - could impact the performance of a very busy database
        - could use a read replica to avoid these issues
      2. stream the database changes from your source relational database redo log

     AWS Glue workflow: Orchestrate repeatable data pipelines
       - easy way to create and visualize your business transformation rules
       - allow for parameters and pipeline state to be shared across stages
       - dynamic views allow inspection of current running workflows for diagnotic and current state info


  Security and access control

    Security personas in AWS Lake Formation
       Data Lake Admins
         - run and operate the data lakes
         - define secure storage boundaries
         - manage users
         - audit/optimize data lakes
       Data Lake Users [consumers]
         - create, generate insight, consume and curate data sets
         - configure and manage access controls across data assets [custodians of the datasets]

       Example: Table Permission (by Data Lake Admin)
           -> AWS Lake -> Data Catalog -> Tables -> select Table -> Actions -> Grant ->
              select <user> ->
                 # grantable permissions allows user to grant their permissions to other users
                 Grant Permissions orders:  Table permissions: Select, Grantable Permssions: Select -> Grant
                 # under columns, you can select columns and exclude them from view



     Security Deep Dive

        users
          - principals can be IAM Users, roles, active directory users via federation

                           End-services retrieve
                           underlying data
                           directly from S3
                 1 Query T                    2  request Access T
          Users  ----->    Athena            ------------------------>    AWS Lake Formation
                           EMR
                           Redshift           <---------------------->         S3
                           Glue               3 short-term creds for T

                                              4 request obj comprising T
                                              ------------------------->

                                              <-----------------------
                                                return object of T

              Steps
                 1. Query T
                 2. Athena queries AWS Lake for permissions - request access T
                 3. returns temporary credentials along with IAM policy of the users level of access
                 4. Athena uses the temporary creditials and IAM policy to access the results sets
                 5. Object return with any filter colomns removed


  Data discovery and collaboration
     Tags
       - add tags to tables so you can filter based on tags

     Visualising forecasts
       Athena
         - Serverless interactive query
         Hive
           - Use the supported data definition language (DDL) statements presented here directly in Athena.
           - The Athena query engine is based in part on HiveQL DDL
           Used for DDL (data definition language) functionality
             - complex data types
             - multitude of formats
             - supports data partitioning (e.g. CREATE TABLE, ALTER TABLE, MSCK REPAIR)
         Presto (SQL engine version 2) and Trino  (SQL engine version 3)
           - Athena for SQL uses Trino with full standard SQL support and works with various standard data formats,
             including CSV, JSON, ORC, Avro, and Parquet.
           - Athena for SQL uses Trino and Presto with full standard SQL support and works with various standard
             data formats, including CSV, JSON, Apache ORC, Apache Parquet, and Apache Avro
           Used for SQL functionality
             - in-memory distributed query engine
             - ANSI-SQL compatible with extensions (e.g. SELECT * FROM tableName)

       QuickSight
         - interactive dashboards
         - add rich ineractivity like filters, drill downs, zooming, and more
         - fast navigation
         - Accessible on any device
         - data refresh
         - Publish to everyone with a click

       Data visualization with Athena via ODBC/JDBC connector
         - Amazon Quicksight
         - tableau
         - looker
         - Qlik
         - Power BI

  Auditing and monitoring
    AWS Lake Formation -> Dashboard (as Lake Admin) -> Recent access Activity (center bottom pane)
      - lists all actions that have occurred on your data lakes

    CloudTrail
       - access events are also published in cloudtrail
       - cloudWatch logs can be integrate with CloudTrail to report on access events



###################################################
AWS Lake Formation — Step by Step guide
  Medium
  https://learning-dipali.medium.com/aws-lake-formation-step-by-step-guide-701661de4be

  Summary:
    In this tutorial, we will create a data lake using AWS Lake Formation.
    Here we will ingest batch data as well as real-time data in our data lake.

  Github:
    https://github.com/dipalikulshrestha/datalakeformation

   stopped at: Run the Crawler : Realtimecrawler
###################################################
AWS Lake Formation FAQs
https://aws.amazon.com/lake-formation/faqs/


What is AWS Lake Formation?
  - AWS Lake Formation makes it easier to centrally govern, secure, and globally share data for analytics and
    machine learning (ML).
  - With Lake Formation, you can centralize data security and governance using the AWS Glue Data Catalog, letting
    you manage metadata and data permissions in one place with familiar database-style features.
  - It also delivers fine-grained data access control, so you can help ensure users have access to the right data,
    down to the row and column level.


How does Lake Formation relate to AWS Glue and the AWS Glue Data Catalog?
  - Lake Formation shares console controls and the AWS Glue Data Catalog with AWS Glue. AWS Glue focuses on
    data integration and ETL.


Can I use third-party business intelligence tools with Lake Formation?
  - Yes. You can use third-party business applications, such as Tableau and Looker, to connect to your AWS data
    sources through services such as Amazon Athena or Amazon Redshift.
  - Access to data is managed by Lake Formation and the underlying AWS Glue Data Catalog, so regardless of which
    application you use, you’re assured that access to your data is governed and controlled.

###################################################
AWS Lake Formation — Step by Step guide
  Medium
  https://learning-dipali.medium.com/aws-lake-formation-step-by-step-guide-701661de4be

  Summary:
    In this tutorial, we will create a data lake using AWS Lake Formation.
    Here we will ingest batch data as well as real-time data in our data lake.

  Github:
    https://github.com/dipalikulshrestha/datalakeformation

   stopped at: Run the Crawler : Realtimecrawler

