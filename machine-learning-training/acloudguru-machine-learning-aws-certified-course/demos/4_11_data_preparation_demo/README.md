4.11 Demo: Data Preparation


  Note: Downloaded demo files to:
      C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\4_11_data_preparation_demo


  From previously collected data, you need to find the following:
    - which percentage of users are male vs female?
    - What are the sages of most of the users?
    - of the users, how many are in their 20s, 30s, 40s, etc.?
    - Convert data from JSON to CSV and store it in S3
    - Transform 'gender' feature to binary value (e.g. male to 1, female to 0)


  Answering Business Questions:
    - Use Athena to query the data collected to answer the business questions
    - CSV format stored in S3 containing First Name, Last Name, Age, Gender (male: 1, female: 0), Longitude, Latitude

  Transform data:
    - Setup an AWS Glue job to run Apache Spark code (or Python shell code or Scala code) to transform
      the data and convert it to CSV formatto transform data and convert to CSV format
    - also apply the mapping to the gender attribute with this AWS Glue job


   AWS Glue & Athena:
      S3  --> Crawler   --->   Data Catalog          --->  Athena
                               Database Tables
                                                     ---> AWS Glue (python and Apache Spark)  --> S3




  # create a AWS Glue crawler to search S3 bucket
  AWS Console  -> AWS Glue --> Data Catalog (left tab) -> Tables -> Add tables using Crawler ->
      Crawler Details: Name: my-user-data-crawler -> Next
      Data Source -> Add a data source: Data Source: S3, S3 path: s3://my-prod-userdata -> Add an S3 data source
      -> Next -> Create New IAM Role: Name: AWSGlueServiceRole-UserDataRole  ->  Create
      -> Next -> Output Configuration:
                     Target Database -> Add Database: Database Details: Name: my-user-database -> Create database
                Target Database: my-user-database
                Crawler Schedule: Frequency: On demand -> Next  -> Create Crawler
                -> Run Crawler
                -> Go to AWS GLue -> Table -> my_prod_userdata -> shows table with columns from files (e.g. geneder, name,
                   location, email, dob, ....) along with data types
                   Also create colunms with names "partition_0" ... "partition_3" as way to separate various folders in bucket

  # use Athena to query data
  AWS Console  -> AWS Athena -->  Query editor (left tab) -> Now shows Data Source: AwsDataCatalog , Database: my-user-database
     -> Tables (left side) -> right click on 3-dots, select "Preview table"
       -> this place the following doe in Query 1: SELECT * FROM "my-user-database"."my_prod_userdata" limit 10;
       -> Run
       -> Create Query 2: to have specific columns:
          old: SELECT "name", "gender", "dob", "location" FROM "my-user-database"."my_prod_userdata" limit 10;
          new: SELECT "name"."first", "name"."last", "dob"."age", "gender", "location"."coordinates"."latitude",
                 "location"."coordinates"."latitude" FROM "my-user-database"."my_prod_userdata" limit 10;
       -> Create Query 3:  report number of records in S3 data:
       SELECT count(*) FROM "my-user-database"."my_prod_userdata"

       -> Create Query 4: percentage of male and female
       -> Create Query 5 : top 5 ages
         New:
          SELECT "dob"."age", COUNT("dob"."age") AS occurances
          FROM my_prod_userdata
          GROUP BY "dob"."age"
          ORDER BY occurances DESC
          LIMIT 5;

       -> Create Query 5 : group ages to 5 bins:
          New:
            SELECT SUM(CASE WHEN "dob"."age" BETWEEN 21 AND 29 THEN 1 ELSE 0 END) AS "21-29",
            SUM(CASE WHEN "dob"."age" BETWEEN 30 AND 39 THEN 1 ELSE 0 END) AS "30-39",
            SUM(CASE WHEN "dob"."age" BETWEEN 40 AND 49 THEN 1 ELSE 0 END) AS "40-49",
            SUM(CASE WHEN "dob"."age" BETWEEN 50 AND 59 THEN 1 ELSE 0 END) AS "50-59",
            SUM(CASE WHEN "dob"."age" BETWEEN 60 AND 69 THEN 1 ELSE 0 END) AS "60-69",
            SUM(CASE WHEN "dob"."age" BETWEEN 70 AND 79 THEN 1 ELSE 0 END) AS "70-79"
            FROM my_prod_userdata;

  # use AWS Glue to query data
  # Create an ETL job to transform data to CSV and converts gender attribute from male/female to 1/0


  # first create output bucket
  AWS Console -> S3 -> bucket name: my-userdata-glue -> Create Bucket
  arn: arn:aws:s3:::my-userdata-glue

  # next UserDataRole-xxxx-Policy attached to UserDataRole  to include new bucket
  AWS Console -> S3 -> bucket name: my-userdata-glue -> Create Bucket

  updated policy:
     {
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": [
				"s3:GetObject",
				"s3:PutObject"
			],
			"Resource": [
				"arn:aws:s3:::my-prod-userdata*",
				"arn:aws:s3:::my-userdata-glue"
			]
		}
	]
     }

  AWS Console  -> AWS Glue -->  ETL jobs (left tab) -> select "Visual ETL" ->  Start Fresh, Engine: Spark -> Create Script
     -> Job Details <tab>  -> Name: my-userdata-transformation-job,  Type: Spark, Language: Python3, IAM Role: xxx-UserDataRole

     -> PROVIDED AWS GLUE LAB flow does not work in the current AWS GLUE environment



