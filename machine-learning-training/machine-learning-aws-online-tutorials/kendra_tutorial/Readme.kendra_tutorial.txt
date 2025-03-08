-----------------------------------------
  AWS Kendra
-----------------------------------------

AWS Kendra - Enterprise Search Service | Create Index, Custom Datasource & Search Experience

JasCloudTech
https://www.youtube.com/watch?v=QqLE_8mJCR8


Summary:
 Premiered Apr 16, 2024
Amazon Kendra, an intelligent enterprise search service, to traditional search solutions.
Amazon Kendra's ML models, accuracy, and ease of use make it easier for customers and employees to find the
information they need when they need it

Introduction

  Kendra
    - enterprise search service using ML models

Basic Knowledge

   Kendra workflow
     Step 1: Create Index
       - create an index where you will add your data sources

     Step 2: Add data sources
       - Use Kendra's connectors for popular sources like file systems, web sites, Box, DropBox,
         Salesforce, SharePoint, relational databases, and S3

     Step 3: Test and Deploy
       - Test and tune the search experience directly in the console
       - Access sample code for each component of the search experience so you can easily deploy to
         new or existing applications

Prerequiste

  Add data to be index to S3 bucket
    -> Add IT help desk PDF files to jon-demo-bkt
       Upload ->
       troubleshooting-guide-for-service-desk-teams.pdf
          downloaded from:
          https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.zscaler.com/resources/ebooks/troubleshooting-guide-for-service-desk-teams.pdf&ved=2ahUKEwjNg9zt9u6LAxWiOTQIHTQLEJAQFnoECBMQAQ&usg=AOvVaw0vAeFI5dP0DnNk8FyXLf5B
       Common-Computer-Issues-and-Solutions.pdf
          downloaded from:
            https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.linnmar.k12.ia.us/wp-content/uploads/2020/08/Common-Computer-Issues-and-Solutions.pdf&ved=2ahUKEwj6s9Of9-6LAxU-ATQIHT9nAM0QFnoECBUQAQ&usg=AOvVaw0ANRHxMKdDsRyDvJ7V6WUQ


Create Index

  Kendra Index Access control setting:
  Use tokens for access control?
    - Choose whether to implement token-based access control for this index. You can change this later.
   No
     - Choose this option to make all indexed content searchable and displayable for all users.
     - Any access control list is ignored, but you can filter on user and group attributes.
   Yes
     - Choose this option to enable token-based user access control.
     - All documents with no access control and documents accessible to the user are searchable
       and displayable.


   Editions
     - Choose an edition
     - For information on our free tier, document size limits, and total storage for each Kendra edition,
       please see our pricing page. Pricing information
     Developer edition
       - Use this edition to test Amazon Kendra's capabilities and build a proof-of-concept search based
         application in a development environment.
           Storage capacity: - Up to 10,000 documents
           Query capacity: - 4,000 queries/day
           Availability zones (AZ) 1

     Enterprise edition
       - Use this edition to implement a production ready Amazon Kendra search index.
           Storage capacity: 100,000 - 10,000,000+ documents
           Query capacity: 8,000 - 800,000 queries/day
           Availability zones (AZ): 3

  AWS -> Kendra -> Create Index ->
    Index Name: mykendraindexdemo,
    IAM : Create a new role, Role Name: AmazonKendra-us-east-1-demokendraindex
    -> Next ->
    Edition: Developer edition
    -> Next -> Review -> Create
    # takes ~30 sec


Add Custom Datasource


  Kendra Data Source:
    - A data source contains a connector that establishes a connection between the documents stored in your
      repository and your Amazon Kendra index.
    - The data source uses the connector to watch your document repository and determine which documents
      need to be indexed, re-indexed, or deleted.
    - Use Kendra's connectors for popular sources like file systems, web sites, Box, DropBox, Salesforce,
      SharePoint, relational databases, and S3, and more

      Note: Kendra data source includes Option for sample S3 Data source (ocvers kendra, EC2, S3, and Lambda)

  Kendra Data source Sync mode
    - Choose how you want to update your index when your data source content changes.
    Full sync
      - Sync and index all contents in all entities, regardless of the previous sync status.
    New, modified, or deleted content sync
      - Only sync new, modified, or deleted content.

  Kendra Data Source Sync run schedule
    - Tell Amazon Kendra how often it should sync this data source.
    - You can check the health of your sync jobs in the data source details page once the data source is
      created.
    Frequency
       Run on Demand, hourly, daily, weekly, monthly, or custom


  AWS -> Kendra -> Indexes -> mykendraindexdemo -> Add Data Source ->
     Available Data sources -> Amazon S3 connector -> Add connector ->
       Data source name: jon-demo-bkt -> Next ->
       IAM : Create a new role, Role Name: AmazonKendra-demokendradatasource
       -> Next ->
        Enter the data source location:  s3://jon-demo-bkt
        max file size <default> 50 MB
        Sync Mode: Full sync
        Sync Schedule: Run on demand
        -> Next -> set Field Mappings - optional
        -> Next -> Review -> Add Data source
        # crawl on your S3 data source
        -> sync Now
        # for 1 doc, ~5 min
        -> examine sync history

  # test search
  AWS -> Kendra -> Indexes -> mykendraindexdemo -> Search Index content <left tab>

      search: computer running slowly
        -> shows content from index docs

User [create] Experience


   # need to create a Identity Center user first
   IAM Identity Center -> Users -> Create User ->
      username: <userName>
      email: <email account>
      generate a one-time password
      First Name: Jon
      Last Name: Smith
       -> Add user

  AWS -> Kendra -> Indexes -> mykendraindexdemo -> Experiences <left tab> -> Create Experience ->
    Experience Name: mydemoexp, Data Sources: jon-demo-bkt
       IAM : Create a new role, Role Name: AmazonKendra-demokendraexperience
       -> Next ->
       Users: JonSmith
       -> Create Experience
       # after created

       -> Open Experience Builder


     -> Log in to Identity Account:
        https://d-9067d50c4f.awsapps.com/start
        user: jonsmith
        pw

        AWS access portal -> Applications
           -> click on: kendra Experience builder app
              search: slow computer


    Clean-up

      AwS -> Kendra -L indexes -> mykendraindexdemo -> delete

