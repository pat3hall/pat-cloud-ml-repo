------------------------------------------------------

Building a Leaderboard with Amazon Managed Service for Apache Flink | 1/5
https://www.youtube.com/watch?v=ZrybveqaWls

  See files in: tutorials/aws_building_leaderboard_tutorial

  Resources:
  Github repository:
  https://github.com/build-on-aws/real-time-gaming-leaderboard-apache-flink

  Resources used in this video:
  🔗 Overview of AWS CDK (Cloud Development Kit):
  https://docs.aws.amazon.com/cdk/v2/guide/home.html
  https://docs.aws.amazon.com/cdk/v2/gu...
  🔗 Using a Studio notebook with Managed Service for Apache Flink:
  https://docs.aws.amazon.com/managed-flink/latest/java/how-notebook.html
  🔗 Apache Flink SQL API:
  https://nightlies.apache.org/flink/flink-docs-release-1.18/docs/dev/table/sql/gettingstarted/
  🔗 Intro to Amazon Managed Service for Apache Flink:
  https://docs.aws.amazon.com/managed-flink/latest/java/what-is.html
  🔗 Intro to Amazon Kinesis Data Streams:
  https://docs.aws.amazon.com/streams/latest/dev/introduction.html



  Most common streaming use cases:
    - anomaly detection
    - IoT Analytics
    - rapid response or emergency response
    - Real-time personalization
    - Asset tracking (gaming leaderboard)

  Streaming Architecture stages

     Ingestion --> Streaming Storage ---> Enrich/Process/Analyze ---> Result Storage --> Visualization

     Streaming storage:
         Amazon MSK (Managed Streaming for Apache Kafka)
         Kinesis Data Streams (KDS)
     Enrich/Process/Analyze
       - about applying the business logic
     Result Storage:
       database, data lakes

  Gaming leaderboard use case
     Gather gaming event data from online gaming (e.g. for car racing game: player ID, total distance, speed)

                                 Kinesis
                    -----------> firehose   ---> S3         ----> MSF     ---->  KDS
                    |            archiver        Datalake         Replay         leaderboard
                    |                                                            replay
     Lambda         |            Zepplin
     Lambda         |            notebook
     Gaming  ----> KDS         ---> MSF      ----> KDS        ---> Lambda
     events        leaderboard      ^ ^            Leaderboard     Redis Sync
                   events           | |            results          results
                                    | |                              |
     Lambda   ---> Aurora ----------| |                              V
     Player         MySQL             |                            MemoryDB for Redis
                                      |                             results storage
     Lamda    --->  KDS  -------------|                              |
     Config                                                          V                        Dashboard
     Pusher                                                         EC2             --------> Users
                                                                    Gaming Server
                                                                    (public subnet)


  Pre-requisites

    Latest Node JS and npm
    Latest cdk npm install -g aws-cdk
    Python 3 with pip3

  Deploying

    Take a check out of a main branch.
    Switch to the folder infra/functions/players and run pip3 install -r requirements.txt -t .
    Switch to the folder infra/functions/redis-sync and run pip3 install -r requirements.txt -t .
    Go to the infra folder and run npm install
    Go to the infra folder and run cdk bootstrap  # if using CDK for the first time in the given AWS account and region, else skip this step.
    Go to the infra folder and run cdk deploy

  cdk shutdown stacks:
    cdk destroy [STACKS..]          Destroy the stack(s) named STACKS

cdk deploy:
  - deploys 2 cloudformation stacks
                CDKToolkit
                  - includes ECR::Repository, IAM:Roles, IAM::Policies, S3 Bucket
                GamingLeaderboardStack
                  - includes Kinesis::Stream, Lambda::Functions, Glue::Database, Log::LogGroup, Log::LogStream,
                    KinesisAnalyticsV2::Application. VPC with subnets, IGW, & GW, EC2::SecurityGroups, SecretManage::Secrets,
                    EC2::Instance, MemoryDB::Cluster, and CDK::Metadata

  Open MSF (Kinesis Analytics) studio notebook:

     Kinesis Analytics (Managed Apache Flink) -> Studio Notebooks <left tab> ->  Create Studio notebook ->
       Creation Method: Create with Custom Settings, name: gaming-demo-1 -> Next
        -> IAM: Choose from IAM roles ..., Service Role: GamingLeaderBoardStack-notebookcommonnotbookrole,
        AWS Glue Database: leaderboard  -> Next
        -> Configuration <use defaults> -> Next -> Review -> Create Studio Notebook
        # after notebook is created
        from 'gaming-demo-1' page -> click "Run" <top right>
        # after notebook is started
        click "Open in Apache Zeppelin" -> Import -> real-time-gaming-leaderboard-apache-flink-main\notebooks\challenges.zpln
        -> click "challenges"
        -> in code block 2, change region to "us-east-1"
        -> run 1st two code blocks

------------------------------------------------------
