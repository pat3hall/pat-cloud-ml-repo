
Set up streaming ETL pipelines
with Apache Flink and Amazon Kinesis Data Analytics
Tutorial

https://aws.amazon.com/tutorials/set-up-streaming-etl-pipelines-apache-flink-and-amazon-kinesis-data-analytics/

n this tutorial you will learn how to:

    Create an Amazon Kinesis Data Stream
    Set up an Amazon Kinesis Data Generator
    Send sample data to a Kinesis Data Stream
    Create an Amazon S3 bucket
    Download code for a Kinesis Data Analytics application
    Modify application code
    Compile application code
    Upload Apache Flink Streaming Java code to S3
    Create, configure, and launch a Kinesis Data Analytics application
    Verify results
    Clean up resources

 Note: Tutorial uses 'mvn' (Apache Maven) so it must be installed along
     with JDK (Java). 'mvn' is used to compile the Kinesis Java program

     Installing Maven
       https://maven.apache.org/install.html

     Installing JDK
       https://docs.oracle.com/en/java/javase/11/install/installation-jdk-microsoft-windows-platforms.html


Amazon Kinesis Data Generator
  - how to setup KDG
  https://awslabs.github.io/amazon-kinesis-data-generator/web/help.html

  Username: jsmith
       Note: password did not allow special character '@'.
             Looks like only alpha numeric characters are allowed
         a


Step 6.1 compiling the S3Sink java code failed:

[ERROR] Failed to execute goal on project aws-kinesis-analytics-java-apps: Could not resolve dependencies for project com.amazonaws:aws-kinesis-analytics-java-ap
ps:jar:1.0
[ERROR] dependency: org.apache.flink:flink-connector-kinesis:jar:1.11.1 (compile)
[ERROR]         org.apache.flink:flink-connector-kinesis:jar:1.11.1 was not found in https://repo.maven.apache.org/maven2 during a previous attempt. This failure
 was cached in the local repository and resolution is not reattempted until the update interval of central has elapsed or updates are forced
[ERROR] dependency: org.apache.flink:flink-streaming-java:jar:1.11.1 (provided)
[ERROR]         org.apache.flink:flink-streaming-java:jar:1.11.1 was not found in https://repo.maven.apache.org/maven2 during a previous attempt. This failure wa
s cached in the local repository and resolution is not reattempted until the update interval of central has elapsed or updates are forced



