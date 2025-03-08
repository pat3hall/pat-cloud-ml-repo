
This directory contains:

Empowering SQL Developers with Real-Time Data Analytics and Apache Flink
AWS Tutorial
  https://community.aws/content/2g3glfVJ0Fi63UWcMqtoNhjqI5y/real-time-analytics-with-flink-sql


The proposed solution consists of the following elements:

    Our sample data is NYC Taxi Cab Trips data set that includes fields capturing pick-up and 
    drop-off dates/times/locations, trip distances and more. This will behave as a streaming data. 
    However, this data does not contain information which borough the pick-up and drop-off belong 
    to, therefore we are going to enrich this data with a Taxi Zone Geohash data provided in another file.

      - Ingest the data from S3 to KDS via MSF Studio Notebooks
      - Analyse data in KDS with SQL via MSF Studio Notebooks
      - Write processed data to S3
      - Build and deploy the MSF application


  NYC Taxi Cab Trip dataset
     https://sharkech-public.s3.amazonaws.com/flink-on-kda/yellow_tripdata_2020-01_noHeader.csv

  Git Repo:
    https://github.com/build-on-aws/real-time-analytics-with-flink-sql/tree/main


   MSF new role name: kinesis-analytics-flink-sql-notebook-us-east-1
