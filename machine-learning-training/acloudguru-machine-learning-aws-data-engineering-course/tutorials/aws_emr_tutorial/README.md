------------------------------------------------------

  AWS EMR Tutorial - by Johnny Chivers:
    https://www.youtube.com/watch?v=v9nk6mVxJDU

    see files in: acloudguru-machine-learning-aws-data-engineering-course\tutorials\aws_emr_tutorial

    Slides:
      https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-overview.html
      -> PDF download in aws_emr_tutorial directory

    Doing AWS ETL on Amazon Workshop booklet
    https://catalog.us-east-1.prod.workshops.aws/workshops/c86bd131-f6bf-4e8f-b798-58fd450d3c44/en-US/setup

    setup:
       1. create vpc
       Create a VPC -> "VPC and more", tag: "emr-tutorial", IPv4: 10.0.0.0/16, No of AZs: 2, public subnets: 2,
         private subments: 0, NAT gateway: None, VPC Endpoints: S3 Gateway
          -> Create VPC

       2. create cloud9 IDE
         -> Cloud9 IDE not available to new customers after 7/25/24

       4. create  Key pair
         EC2 -> Key pairs -> Create Key Pair -> Name: emr-tutorial, RSA, PEM -> create

         chmod 400 emr-tutorial.pem

       5. download git repo
          download and unzip



  EMR (Elastic MapReduce)
    - Managed cluster platform that simplifies running big data frameworks
    - data processsing framework (or it contains data processing framework)
    - on the cluster, there are different data proceessing frameworks that can be setup (e.g. spark, hive, pig)

    Master node:
      - A node that manages the cluster by running software components to coordinate the distribution of
        data and tasks among other nodes
    Core node:
      - A node with software components that run tasks and store data in the Hadoop Distributed File System
        (HDFS) on your cluster.
      - data is stored on core nodes in HDFS or on S3 using EMRFS
      - Multi-node clusters have at least one core node.
    Task node (optional):
      - A node with software components that only runs tasks and does not store data in HDFS.

    Note:
      - must have at least 1 core node and 1 master node for a multi-node cluster

    Data Processing Frameworks
      - Engine used to process and analyze data
      - Different frameworks are available for different kinds of processing needs
      - The main processing frameworks available for Amazon EMR are Hadoop MapReduce and Spark
    Storage
      - Hadoop Distributed File System (HDFS) is adistributed, scalable file system for Hadoop.
      - Using the EMR File System (EMRFS), Amazon EMR extends Hadoop to add the ability to directly
        access data stored in Amazon S3 as if it were a file systemlike HDFS. (most common approach)
      - The local file system refers to a locally connected disk. (just small disks used for OS, etc)
    Cluster Resource Management
      - The resource management layer is responsible for managing cluster resources and scheduling the jobs
        for processing data.
      - By default, Amazon EMR uses YARN (Yet Another Resource Negotiator).

  Spark ETL

    What is Spark?
      - Apache Spark™ is a multi-language engine forexecuting data engineering, data science, and machine
        learning on single-node machines orclusters.
    Faster Processing
      - Spark contains Resilient Distributed Dataset(RDD) which saves time in reading and writing operations,
        allowing it to run almost ten to onehundred times faster than Hadoop with MapReduce.
    In Memory Computing
      - Spark stores the data in the RAM of servers which allows quick access and in turn accelerates the
        speed of analytics.
      - Note: MapReduce stores data on disk and then reduces it together in memory pulling it writes back
        to disk over several steps. With Spark, it tries to keep as much of the data in memory for faster
        processing
    Flexibility
      - Apache Spark supports multiple languages and allows the developers to write applicationsin Java,
        Scala, R, or Python.


  Hive
    What is Hive?
      - The Apache Hive ™ data warehouse software facilitates reading, writing, and managing large
        datasets residing in distributed storage usingSQL.
      - Structure can be projected onto data already in storage.
      - A command line tool and JDBC driver are provided to connect users to Hive.
    SQL Like Interface
      - Hive provides the necessary SQL abstraction to integrate SQL-like queries (HiveQL) into the
        underlying Java without the need to implement queries in the low-level Java API.
    Storage
      - Different storage types such as plain text, RCFile , HBase , ORC, and others.

  PIG
    What is PIG?
      - Apache Pig is a platform for analyzing large data sets that consists of a high-level language
        for expressing data analysis programs, coupled with infrastructure for evaluating these programs.
      - The salient property of Pig programs is that their structure is amenable to substantial
        parallelization, which in turns enables them to handle very large data sets.
    How Does It Work?
      - It is an abstraction of Map Reduce which integrates with the lower level java api which means
        parallel processing is easily achieved.
    Storage
      - Different storage types such as plain text, RCFile , HBase , ORC, and others.
    Note:
      - PIG is going out of fashion slightly because it is slower that spark but it is still used
      - provided parallelization of your data without writing low-level java code (presumably compared to MapReduce)

  AWS Step Functions

    What Are Step Functions?
      - AWS Step Functions is a low-code, visual workflow service that developers use to build distributed
        applications, automate IT and business processes, and build data and machine learning pipelines
        using AWS services.
      - Workflows manage failures, retries, parallelization, service integrations, and observability so
        developers can focus on higher-value business logic.

    EMR Autoscaling
      How does it work?
        - Autoscaling policies are added to an EMR cluster which define how nodes should be added or removed.
        - There options in terms of available RAM, disc, app running, apps pending etc.



   AWS Console -> EMR -> Create Cluster ->

   name: emr-tutorial-cluster1,
   EMR release: emr-5.35.0,
   Application bundle: Hadoop 2.10.1, Hive 2.3.9, Hue 4.10.0, JupyterEnterpriseGateway 2.1.0, JupyterHub 1.4.1,
       Livy 0.7.1, Pig 0.17.0, Spark 2.4.8
       vpc: emr-tutorial-vpc,
       cluster composition: Uniform instance groups
       remove task node instance group
       Primary instance type: m5.xlarge
       core instance type: m5.xlarge
       core node: 2
       key pair: emr-tutorial
       create service role
       create instance profile, All S3 buckets ... read and write access"
       -> create cluster

     -> primary node -> Security Group
         -> add rule:  SSH (port 22) from: MyIP -> add rule
         -> add rule:  Custom TcP port 9443 from: MyIP -> add rule

     -> EMR -> Clusters -> emr-tutorial-cluster1 -> Summary (top) -> Cluster Management
        -> Connect to the Primary node using SSH
        ssh -i ~/emr-tutorial.pem hadoop@ec2-54-234-27-130.compute-1.amazonaws.com


   # create S3 bucket for tutorial
   AWS Console -> S3 -> Create Bucket -> Name: emr-tutorial-bkt-ph -> create bucket
      -> Create folders -> input, output, files, logs
      -> updoad "tripdata.csv" to "input" folder


   # ssh on to EMR primary node
        ssh -i ./emr-tutorial.pem hadoop@ec2-54-234-27-130.compute-1.amazonaws.com
        # create file "spark-etl.py" from provided github file
        vi spark-etl.ph

        # run the following commands from "sparkCommands.txt" with <YOUR-BUCKET> replaced with S3 bucket name
        export PATH=$PATH:/etc/hadoop/conf:/etc/hive/conf:/usr/lib/hadoop-lzo/lib/:/usr/share/aws/aws-java-sdk/:/usr/share/aws/emr/emrfs/conf:/usr/share/aws/emr/emrfs/lib/:/usr/share/aws/emr/emrfs/auxlib/

         export PATH=$PATH:spark.driver.extraClassPath/etc/hadoop/conf:/etc/hive/conf:/usr/lib/hadoop-lzo/lib/:/usr/share/aws/aws-java-sdk/:/usr/share/aws/emr/emrfs/conf:/usr/share/aws/emr/emrfs/lib/:/usr/share/aws/emr/emrfs/auxlib/

        spark-submit spark-etl.py s3://emr-tutorial-bkt-ph/input/ s3://emr-tutorial-bkt-ph/output/spark

        -> runs -> reads in tripdata.csv, adds "current_date" column , writes parquet output file to S3

       # examine spark

     -> EMR -> Clusters -> emr-tutorial-cluster1 -> Summary (top) -> Cluster Management
        -> click on "Spark History Server" -> wait ~5 min for it to update

     -> Go to Primary Node Jupyter Notebook website (9443)
        https://<primaryNodePublicIP>:9443
           -> can type in browers "this is unsafe"  or click "Advanced" -> accept risks

           # from AWS EMR tutorial doc
           Username: jovyan
           password: jupyter

           -> create new notebook -> click "new" type: PySpark
              -> copy in notebook steps from "JupyterHubNotebookCode.py
              -> similar to
              -> runs:  reads in tripdata.csv, printSchema, adds "current_date" column , printSchema,


        Code: JupyterHubNotebookCode.py PySpark code
        >>> # Step 1
        >>> import sys
        >>> from datetime import datetime
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.functions import *
        >>>
        >>> # Step 2
        >>> input_path = "s3://emr-tutorial-bkt-ph/input/tripdata.csv"
        >>> output_path = "s3://emr-tutorial-bkt-ph/output/"
        >>>
        >>> # Step 3
        >>> nyTaxi = spark.read.option("inferSchema", "true").option("header", "true").csv(input_path)
        >>>
        >>> # Step 4
        >>> nyTaxi.count()
        >>>
        >>> # Step 5
        >>> nyTaxi.show()
        >>>
        >>> # Step 6
        >>> nyTaxi.printSchema()
        >>>
        >>> # Step 7
        >>> updatedNYTaxi = nyTaxi

    EMR Step section of Tutorial

      -> upload "emrZeroToHero-main/sparkEmrSteps/spark-etl.py" to "files" folder in "emr-tutorial-bkt-ph"
         -> same as previous "spark-etl.py" except parquet write includes .mode("overwrite")


     -> EMR -> Clusters -> emr-tutorial-cluster1 -> Step (tab) -> Add Step
        Type: Custom JAR
        Name: Custom JAT, JAR location: command-runner.jar,
        Arguments: spark-submit s3://emr-tutorial-bkt-ph/files/spark-etl.py s3://emr-tutorial-bkt-ph/input s3://emr-tutorial-bkt-ph/output
        -> add step
        -> wait for step "completed"
        note: need to wait a few minutes after "completed" to view logs
        -> view logs -> stdout

   Hive section of Tutorial - Using Hive CLI

   # ssh on to EMR primary node
        ssh -i ./emr-tutorial.pem hadoop@ec2-54-234-27-130.compute-1.amazonaws.com

        # log into Hive
        $ hive
        # copy "emrZeroToHero-main/hive/ny_taxi.hql", after changing <YOUR-BUCKET> to bucket name, to hive prompt
        # this will create a table

        hive> CREATE EXTERNAL TABLE ny_taxi_test (
                            vendor_id int,
                            lpep_pickup_datetime string,
                            lpep_dropoff_datetime string,
                            store_and_fwd_flag string,
                            rate_code_id smallint,
                            pu_location_id int,
                            do_location_id int,
                            passenger_count int,
                            trip_distance double,
                            fare_amount double,
                            mta_tax double,
                            tip_amount double,
                            tolls_amount double,
                            ehail_fee double,
                            improvement_surcharge double,
                            total_amount double,
                            payment_type smallint,
                            trip_type smallint
                     )
                     ROW FORMAT DELIMITED
                     FIELDS TERMINATED BY ','
                     LINES TERMINATED BY '\n'
                     STORED AS TEXTFILE
                     LOCATION "s3://emr-tutorial-bkt-ph/input/";

           # return the unique rate_code_id values
           hive> SELECT DISTINCT rate_code_id FROM ny_taxi_test;

           ----------------------------------------------------------------------------------------------
                   VERTICES      MODE        STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED
           ----------------------------------------------------------------------------------------------
           Map 1 .......... container     SUCCEEDED      1          1        0        0       0       0
           Reducer 2 ...... container     SUCCEEDED      2          2        0        0       0       0
           ----------------------------------------------------------------------------------------------
           VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 9.84 s
           ----------------------------------------------------------------------------------------------
           OK
           1
           2
           4
           NULL
           3
           5
           Time taken: 17.304 seconds, Fetched: 6 row(s)


   Hive section of Tutorial - EMR steps

      -> upload "emrZeroToHero-main/hive/ny-taxi.hql" to "files" folder in "emr-tutorial-bkt-ph"

     -> EMR -> Clusters -> emr-tutorial-cluster1 -> Step (tab) -> Add Step
        Type: Hive Progam
        Name: Hive Program
        Hive Script location: s3://emr-tutorial-bkt-ph/files/ny-taxi.hql
        input location: s3://emr-tutorial-bkt-ph/input/
        output location: s3://emr-tutorial-bkt-ph/output/hive/
        Arguments:
        -> add step
        -> wait for step "completed"
        note: need to wait a few minutes after "completed" to view logs
        -> Check S3 bucket for "output/hive/00000_0" file

   PIG section of Tutorial -

      -> upload "emrZeroToHero-main/pig/ny-taxi.pig" to "files" folder in "emr-tutorial-bkt-ph"

     -> EMR -> Clusters -> emr-tutorial-cluster1 -> Step (tab) -> Add Step
        Type: Pig Progam
        Name: Pig Program
        Hive Script location: s3://emr-tutorial-bkt-ph/files/ny-taxi.hql
        input location: s3://emr-tutorial-bkt-ph/input/
        output location: s3://emr-tutorial-bkt-ph/output/pig/
        Arguments:
        -> add step
        -> wait for step "completed"
        note: need to wait a few minutes after "completed" to view logs
        -> Check S3 bucket for "output/pig/part-v000-o000-r-00000" CSV file

   EMR Notebook and SageMaker section of Tutorial -
      -> skipped in video, but notebook provided at:  EMRSparkNotebook.ipynb

   Orchestrating Amazon EMR with AWS StepFunctions section of Tutorial
     -> will add the following steps: spark step, hive step,  pig step, run steps

     -> AWS Console -> Step Functions -> State machines <left tab> -> Create state machine ->
        Blank -> Select ->
        Config <tab> -> Name: EMR-tutorial-steps Type: Standard, Execution Role: Create new role ->

        Code <tab> -> replace code with "sfn.json" code
         -> visualization shows:
                                   Start
                Add: Spark Step to EMR Cluster
                Add: Hive Step to EMR Cluster
                Add: Pig Step to EMR Cluster
                Choice: Terminate Clusters?
                default                     $deleteCluster == true
                  Pass State: Wrap Up           Disable TErmination Protection
                                                Terminate EMR cluster
                                                Pass State: Wrap Up
                                   END


       Create <top left> -> Confirm

       click "start execution"
         Name: emr-tutorial-steps-execution
         input -> import "args.json" but first change: <S3_BUCKET_NAME> to emr-tutorial-bkt-ph"
                                            change: <CLUSTER_ID> to "j-3K10A0CX6T68N"
          -> start execution
          -> wait for execution to be completed

   EMR Autoscaling section of Tutorial -

     -> EMR -> Clusters -> emr-tutorial-cluster1 -> Step (tab) ->  concurrent step: 5   (default:1)

     -> EMR -> Clusters -> emr-tutorial-cluster1 -> hardware (tab) ->
        "edit cluster scaling options" -> Custer scaling: custom scaling: min: 2, max 5,
          select "scale out" -> Rule name: addNode, Add: 1 instance if "AppsRunning" greater than or equal to 2 count for 1 five min period,
          cooldown 60 sec

          select "scale in" -> Rule name: removeNode, Add: 1 instance if "AppsRunning" less than 2 count for 1 five min period,
          cooldown 60 sec



   Setup using CloudFormation Stack
      CloudFormation stack template:
         https://aws-data-analytics-workshops.s3.amazonaws.com/emr-dev-exp-workshop/cfn/emr-dev-exp-self-paced.template


   Cluster Create section of Tutorial - redo based on doc (instead of Video)

   AWS Console -> EMR -> Create Cluster ->

   name: emr-tutorial-cluster-2-1,
   EMR release: emr-6.11.0,
   Application bundle: Hadoop 2.10.1, Hive 3.1.3, Hue 4.11.0, JupyterEnterpriseGateway 2.6.0, JupyterHub 1.4.1,
       Livy 0.7.1, Pig 0.17.0, Spark 3.3.2, Tez 0.10.2
       AWS Glue Data Catalog settings: select "Use for Hive Table metadata" and "Use for Spark Table Metadata"
       vpc: emr-tutorial-vpc,
       cluster compoistion: Uniform instance groups
       remove task node instance group
       Primary instance type: m5.xlarge
       core instance type: m5.xlarge
       core node: 2
       cluster termination and node replacement: Idle time: 05:00:00 (5 hours), no termination protection, unhealthy replacement: turn off
       cluster logs: S3 Location: s3://emr-tutorial-bkt-ph/logs
       key pair: emr-tutorial
       Amazon EMR Service Role: Choose an Existing service role, Service Role: EMRDevExp-EMRClusterServiceRole
       EC2 Instance profile form Amazon EMR: Choose an Existing instance profile, Instance Profile: EMRDevExp-EMR_Restricted_Role
       create instance profile, All S3 buckets ... read and write access"
       -> create cluster

     -> primary node -> Security Group
         -> add rule:  SSH (port 22) from: MyIP -> add rule
         -> add rule:  Custom TcP port 9443 from: MyIP -> add rule

     -> EMR -> Clusters -> emr-tutorial-cluster1 -> Summary (top) -> Cluster Management
        -> Connect to the Primary node using SSH
        ssh -i ~/emr-tutorial.pem hadoop@ec2-54-234-27-130.compute-1.amazonaws.com

   EMR Notebook and SageMaker section of Tutorial -

      Modify IAM Role for EMR Cluster to include SageMaker access

     -> EMR -> Clusters -> emr-tutorial-cluster-2-1 -> Hardware (tab) ->  Core Instance -> Instance ID -> IAM Role (for core EC2 instance)
         -> Add Permission -> AmazonSageMakerFullAccess -> attach

      Create a SageMaker IAM role

        IAM -> Roles -> Create Role ->  Trusted Entity Type: AWS Service, Use Case: SageMaker -> Next
           -> Permissions -> AmazonSageMakerFullAccess -> Next
           -> Role Name: SageMaker-EMR-ExecutionRole  -> Create Role

           -> copy ARN: arn:aws:iam::012345678901:role/SageMaker-EMR-ExecutionRole


      Create an EMR Notebook

      Notes:
        - With the IAM permission set up, you can now create your EMR Notebook.
        - EMR Notebooks are serverless Jupyter notebooks that connect to an EMR cluster using Apache Livy.
        - They come preconfigured with Spark, allowing you to interactively run Spark jobs in a familiar Jupyter environment.
        - The code and visualizations that you create in the notebook are saved durably to S3.

        -> EMR ->  Workspace (Notebooks) -> Create Notebooks  -> Create Studio (required before creating workspace) -> Create Studio
           -> Edit -> add VPC (emr-tutorial-vpc) and Subnet (emr-tutorial_subnet...1b) -> save
           -> Add permissions to workspace role: add EMRFullAccess
           -> attach emr-tutorial-2-1 cluster to workspace
           -> launch workspace
           -> in Jupyter Notebook window -> upload 'EMRSparkNotebook.ipynb"
              -> open EMRSparkNotebook.ipynb
                change: region = 'us-east-1'
                change sagemaker_execution_role = 'arn:aws:iam::012345678901:role/SageMaker-EMR-ExecutionRole'


        -> EMR Studios -> Launch workspace -> Launch with Options ->
           Launch with Jupyter, EMR Cluster: emr-tutorial-cluster-2-1 -> launch Workspace ->



        -> EMR ->  Workspace (Notebooks) -> Create Notebooks  ->  -> Workspace (Notebooks) <bottom pane> -> Attach Cluster
           -> Attach Cluster: "Launch in Jupyter




      Cloudformation Stack created Roles Notes:

         EMRDevExp-EMRClusterServiceRole
            -> Trusted Entities: AWS Service: elasticmapreduce
            -> Permissions: AmazonElasticMapReduceRole (AWS Managed)

         EMRDevExp-EMR_EC2_Restricted_Role
            -> Trusted Entities: AWS Service: ec2
            -> Permissions: EMRDevExp-EMR_EC2_Restricted_Role_Policy  (customer inline)

         EMRDevExp-EMRStudioServiceRole
            -> Trusted Entities: AWS Service: elasticmapreduce
            -> Permissions: EMRDevExp-Studio-Service-Policy (customer inline)

         EMRDevExp-SCLaunchRole
            -> Trusted Entities: AWS Service: elasticmapreduce, and servicecatalog.amazonaws.com
            -> Permissions: SC-Launch-Role-Limited-IAM-Policy (Customer inline)
                            SC-Launch-Role-Policy (Customer inline)

         EMR_DefaultRole
           -> Trusted Entities: AWS Service: elasticmapreduce
           -> Permissions:  AmazonElasticMapReduceRole (AWS managed)

         EMR_EC2_DefaultRole
           -> Trusted Entities: AWS Service: ec2
           -> Permissions:  AmazonElasticMapReduceforEC2Role (AWS managed)

         EMR_Notebooks_DefaultRole
           -> Trusted Entities: AWS Service: elasticmapreduce
           -> Permissions: AmazonElasticMapReduceEditorsRole (AWS managed)
                           AmazonS3FullAccess                (AWS managed)


     Hudi Workshop section of Tutorial
        Hudi

        Apache Hudi
          - an open-source data management framework used to simplify incremental data processing and data pipeline development
            by providing record-level insert, update, upsert, and delete capabilities. Upsert refers to the ability to insert records
            into an existing dataset if they do not already exist or to update them if they do.
          - By efficiently managing how data is laid out in S3, Hudi allows data to be ingested and updated in near real time.
          - Hudi carefully maintains metadata of the actions performed on the dataset to help ensure that the actions are atomic and consistent.

        LAB-COW.ipynb Scope
          - Create Hudi Copy on Write table and perform insert, update and delete operations
          - Create Hudi Merge on Read table and perform insert, update and delete operations
          - Create Hive style partitioning for Copy on Write table

     -> Copy below listed files into HDFS
          # ssh to cluster primary node
          ssh -i ~/xxxx.pem hadoop@<ec2-xx-xxx-xx-xx.us-west-2.compute.amazonaws.com>
          # copy required files to HDFS - Ensure  the below listed files are copied into HDFS.

            $ hdfs dfs -copyFromLocal /usr/lib/hudi/hudi-spark-bundle.jar hdfs:///user/hadoop/
            $ hdfs dfs -copyFromLocal /usr/lib/spark/external/lib/spark-avro.jar hdfs:///user/hadoop/
            $ hdfs dfs -copyFromLocal /usr/share/aws/aws-java-sdk/aws-java-sdk-bundle-1.12.31.jar hdfs:///user/hadoop/


     -> Go to Primary Node Jupyter Notebook website (9443)
        https://<primaryNodePublicIP>:9443
          -> in Jupyter Notebook:
              -> upload "LAB-COW.ipynb"


