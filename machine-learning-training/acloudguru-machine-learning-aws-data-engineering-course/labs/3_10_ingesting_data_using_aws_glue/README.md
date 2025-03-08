------------------------------------------------------
3.10 Ingesting Data Uinsg AwS Glue

About this lab

You are a data engineer tasked with migrating some new CSV files into S3. Once there, you need to add the
schema to Glue using Glue Crawler. The Data Science team would like to train some new models on a combined
table of this information but only using a few select columns. To decrease cost and improve processing time,
you need to create a job in Glue that will combine the CSV files together into one table but only including
the needed columns. Then, you need to run the job and verify success in S3. After that, you will be ready to
inform the Data Science team that the data is ready for training.

Learning objectives
  - Prepare the Environment
  - Create a Glue Crawler
  - Create a Job in Glue

                                                        AWS Glue
                                      |-----------------------------------------------------|
    download   --> S3            ---->| Glue    ---> Data    ---> transform   --->  Save to |  --> S3
    data           Parking Tags       | Crawler      Catalog      drop columns       S3     |
                                      |-----------------------------------------------------|

    ------------------------------------------------------
Solution

Log in to the AWS Management Console using the credentials provided on the lab instructions page.
Prepare the Environment

    In the search bar on top of the console, enter S3.
    From the search results, select S3.
    Click Create bucket.
    On the Create bucket page, for Bucket name, enter a globally unique name, such as bucket followed by random numbers.
    Scroll to the bottom of page and click Create bucket.
    Once created, click on the bucket name.
    Click Create folder.
    On the Create folder page, for the Folder name, enter parking_data.
    Click Create folder.
    Open a new browser window or tab and navigate to the parking ticket dataset.
    Select the DOWNLOAD DATA tab.
    Next to parking-tickets-2022, click DOWNLOAD to download the folder.
    Once downloaded, unzip the folder.
    Return to the browser window or tab showing your S3 bucket.
    Click Upload.
    Click Add files.
    Select Parking_Tags_Data_2022.000.csv from your unzipped parking-tickets-2022 folder.
    Click Open.
    Click Upload. You may need to wait a minute or so for the file to be uploaded.

Create a Glue Crawler

    In the search bar on top of the console, enter glue.
    From the search results, select AWS Glue.
    In the left navigation menu, under Data Catalog, select Crawlers.
    Click Create crawler.
    On the Set crawler properties page, in Name, enter parking.
    Click Next.
    On the Chose data sources and classifiers page, click Add a data source.
    On the Add data source pop-up pane, in Data source: Select S3 from the dropdown menu.
    Under S3 path, click Browse S3.
    Select your parking bucket.
    Select parking-data.
    Select the Parking_Tags_Data_2022.000.csv file.
    Click Choose.
    Once back on the Add data source pane, you may see an error message. You will just need to click somewhere on the screen and the error message will go away.
    Click Add an S3 data source.
    Click Next.
    Click Create new IAM role.
    On the Create new IAM role pop-up pane, under Enter new IAM role, enter parking after the AWSGlueServiceRole-.
    Click Create.
    Click Next.
    Click Add database. This will open a new browser tab.
    On the Create a database page, in Name, enter parking.
    Click Create Database
    Go back to the Glue tab.
    On the Set output and scheduling page, click the refresh icon next to the field under Target database.
    In the dropdown menu under Target database, select the new parking database.
    Click Next
    Click Create crawler.
    Click Run crawler. It may take a few minutes to run.
    In the left navigation menu, under Databases, select Tables.
    On the Tables page, click the refresh icon next to Delete. You should see a new table listed.
    Select the table. You should be able to view the schema for the CSV file.

Create a Job in Glue

    In the left nevigation menu, select Getting started.
    Click Set up roles and users.
    Click Choose roles.
    Select AWSGlueServiceRole-Parking.
    Click Confirm.
    Click Next.
    On the Grant Amazon S3 access page, under Data access permissions select Read and write.
    Click Next.
    Click Next again.
    On the Review and confirm page, click Apply changes.
    Click Author and edit ETL jobs.
    Click Visual ETL .
    In the + Add nodes menu, in the Sources tab, select Amazon S3.
    Select the Amazon S3 node now visible on the canvas.
    In the Data source properties - S3 pane on the right, click Browse S3.
    Select your bucket.
    Select parking-data.
    Select the file parking_data/Parking_Tags_Data_2022.000.csv.
    Click Choose.
    Under Data format, select CSV from the dropdown menu.
    In the top left corner of the canvas, click the + button.
    In the + Add nodes menu, select the Transforms tab.
    Select Drop Fields.
    On the canvas, select the Drop Fields node. It may take a few seconds for the Transform pane on the right to populate with the correct information.
    Under DropFields, check all fields except date_of_infraction, infraction_code, and infraction_description.
    In the top left corner of the canvas, click the + button.
    In the + Add nodes menu, select the Targets tab.
    Select Amazon S3.
    In the canvas, select the new Amazon S3 node.
    In the Data target properties - S3 pane on the right, under Node parents, make sure Drop Fields is selected.
    Under Format, select Parquet from the dropdown menu.
    Under S3 Target Location, click Browse S3.
    Select your bucket.
    Select the radio circle next to parking_data.
    Click Choose.
    Under Data Catalog update options, select Create a table in the Data Catalog and on subsequent runs update schema and add new partitions.
    Under Database, select parking from the dropdown menu.
    Under Table Name, enter results.
    On top of the canvas, replace Untitled job by entering the name Parking.
    Click Save in the upper right corner.
    Click Run. This may take a minute or two to complete.
    In the new green banner, click Run details to see the results of the job.
    In the top left corner, select the hamburger menu icon (the icon that looks like three horizontal lines).
    In the left navigation menu, select Tables.
    Under Tables, select results. You should see under Schema that the table only contains the three columns we did not select.
    In the search bar on top of the console, enter S3.
    From the search results, select S3.
    Select the bucket you created.
    Select the parking-data folder. You should see inside the folder that a new Parquet file has been created.
    ------------------------------------------------------

 parking tickets dataset:
 https://open.toronto.ca/dataset/parking-tickets/

