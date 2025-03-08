Getting started with Pyflink - The Python Interpreter for Apache Flink Interpreter 

https://www.youtube.com/watch?v=00JgwB5vJps


In this video we will showcase how to develop a python flink (pyflink) application locally, then package and deploy the application onto Kinesis Data Analytics for Apache Flink.

Code repo: https://bit.ly/3yZ31bs 

Learn more about Kinesis Data Analytics for Apache Flink: 
https://go.aws/3udFelE



Note: use Below steps to setup virtual env;
-> run from Anaconda Prompt window:
Set up virtual environment for Python using Anaconda
https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/


steps:
1. start Anaconda prompt window
-> in Anaconda prompt window, do:

# create 'flink_python3_8_env" virtual conda env using python 3.8 for anaconda:
$ conda create -n flink_python3_8_env python=3.8 anaconda 

# activate virtual env:
$ conda activate flink_python3_8_env

# prompt should change from '(base)' to '(flink_python3_8_env)'

# verify virtual env, should be from 'flink_python3_8_env' subdirectory:
$ which pip
  /cygdrive/c/Users/pat/anaconda3/envs/flink_python3_8_env/Scripts/pip

# install apache-flink 1.15.2
$ pip install apache-flink==1.15

# create a Jupyter kernel using this virtual env:
$ ipython kernel install --user --name=flink_python3_8_kernel

# verify Juptyper kernel
  Start -> Anaconda -> Jupyter Notebook
    -> change Kernel to "flink_python3_8_kernel
    -> verify: !python --version  # shoud return 3.8

# check java version (need to use java 8 or 11)
ava -version
java version "22.0.2" 2024-07-16
Java(TM) SE Runtime Environment (build 22.0.2+9-70)
Java HotSpot(TM) 64-Bit Server VM (build 22.0.2+9-70, mixed mode, sharing


# downgrade java version (from JDK 22 to JDK 11)

# install Java 11 go to:
https://jdk.java.net/java-se-ri/11-MR3
# click on the download link:
Windows 11/x64 Java Development Kit (sha256) 180 MB
  https://download.java.net/openjdk/jdk11.0.0.2/ri/openjdk-11.0.0.2_windows-x64.zip

unzip "jkd-11.0.0.2" and place at C:\Program Files\Java\

# set Windows Env:
  JAVA_HOME: C:\Program Files\Java\jdk-11.0.0.2
  PATH: add: %JAVA_HOME%\bin

Notes:
# to deactive virtual env:
$ conda deactivate

# to remove` virtual env:
conda remove -n <envName> -all



Getting started section:
https://github.com/aws-samples/pyflink-getting-started/tree/main/getting-started

Install PyCharm IDE:
   https://www.jetbrains.com/pycharm/download/?section=windows
      -> go to bottom of page for "Community Edition" download which is FREE
   -> run pycharm-community-2024.2.1.exe installer
     -> Options: "Create Desktop Shortcut" only -> Next -> Install -> Finish

Start PyCharm
  -> Skip Imports (that is skip "Visual Studio Code)
  -> Get from VCS (Version Control Systems)o
     URL: https://github.com/aws-samples/pyflink-getting-started.git
     Directory: C:\Users\pat\PycharmProjects\pyflink-getting-started
     -> clone
     -> Trust

   # change Interpreter in IDE to point at "flink_python3_8_en
   File -> Settings -> Project "pyflink-getting-started" -> Add interpreter ->
     'select" Conda Environment", select "use existing environment", 
         Use: 'flink_python3_8_env' -> ok -> ok
      

 Set up AWS Resources for local deployment

  -> AWS Console -> Kinesis -> Create Data Stream ->
     Name: input-stream, Capacity mode; on-demand
     -> Create data stream

  -> AWS Console -> Kinesis -> Create Data Stream ->
     Name: output-stream, Capacity mode; on-demand
     -> Create data stream


PyCharm
   -> Project/pyflink-getting-starrted/pyflink-examples/application_properties.json
      -> if streams names are different than "input-stream" and "output-stream" or region 
         is not 'us-east-1", then modify 
   -> Project/pyflink-getting-starrted/pyflink-examples/getting-started.py
      -> right click -> Modify Run Configuration -> 
         # add "IS_LOCAL" "true" env variable
         #  used so it will look for local lib instead instead of flink default location
         "+" Name: IS_LOCAL  value: true -> OK -> Apply -> OK

      -> create subdirectory "lib" 
      #  download the 1.15.2 version of the flink-sql-connector from the Maven Central repo.
        https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kinesis/1.15.2/flink-sql-connector-kinesis-1.15.2.jar
          -> add to 'lib' subdirectoy but dragging and dropping

          -> right-click on flink jar file -> Refactor -> Rename -> copy
            flink-sql-connector-kinesis-1.15.2
            -> verify name matches line 52 


        _
   -> Project/pyflink-getting-starrted/datagen/stock.py
      -> needs boto3
       -> open powershell window
         conda install boto3
         -> Run (generate input data to Kinesis)


   -> Project/pyflink-getting-starrted/pyflink-examples/getting-started.py
         -> Run (create tables and pass input-stream data to output-stream

   -> Project/pyflink-getting-starrted/pyflink-examples/
      zip GettingStarted directory

  -> AWS Console -> S3 -> Create Bucket -> name: pyflink-getting-started
     -> upload "GettingStarted.zip"

  -> AWS Console -> Kinesis Analytics -> Create Data Stream ->
     -> Create streaming application ->  "Create from scratch", Apache Flink version: 1.15,
       Application Name: pyflink-demo, 
       IAM role: Create / update IAM 'kinesis-analytics-pyflink-demo-us-east-1' with required policies
       Templates: Development # to save on cost - uses just 1 Kinesis Processing Unit (KPU)
       -> create data stream

  -> AWS Console -> IAM -> Role -> kinesis-analytics-pyflink-demo-us-east-1

   Give IAM role access:
      S3FullAccess, CloudWatchFullAccess, KinesisFullAccess, CloudWatchLogsFullAccess,
      KinesisAnalyticsFullAccess

  -> AWS Console -> Kinesis Analytics -> pyflink-demo stream -> Config ->  
      Application code location:  S3 bucket: s3://pyflink-getting-started, Path to S3 object: GettingStarted.zip

      Runtime Properties: click "Add new item"
        # values are from "application_properties.json" file
      click "Add new item"
        Group ID: kinesis.analytics.flink.run.options, Key: jarfile,  Value: GettingStarted/lib/flink-sql-connector-kinesis-1.15.2.jar
      click "Add new item"
        Group ID: consumer.config.0, Key: input.stream.name,     Value: input-stream
        Group ID: consumer.config.0, Key: flink.stream.initpos,  Value: LATEST
        Group ID: consumer.config.0, Key: aws.region,            Value: us-east-1
        Group ID: producer.config.0, Key: output.stream.name,    Value: output-stream
        Group ID: producer.config.0, Key: shard.count,           Value: 1
        Group ID: producer.config.0, Key: aws.region,            Value: us-east-1
      click "Add new item"
      -> Save Changes


  # run application
  -> AWS Console -> Kinesis Analytics -> pyflink-demo stream -> Run
         Run without snapshot -> run

  -> PyCharm 
      resume sending data from "stock.py"

  -> AWS Console -> Kinesis Analytics -> pyflink-demo stream -> Open Apache Flink Dashboard
     
