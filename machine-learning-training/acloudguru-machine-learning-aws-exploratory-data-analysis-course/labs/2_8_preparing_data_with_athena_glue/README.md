2.8 Hands-on Lab: Preparing Data Using Amazon Athena and AWS Glue


About this lab

Imagine you are the data engineer and you have been assigned the task to prepare the data and get it ready for the
machine learning engineers to create a highly predictable model. Your corporation has been working with AWS and you
have been encouraged to use AWS services.

Your raw data has been uploaded to an input folder in an S3 bucket. You will use a Glue crawler to detect the schema
structure. You will then upload the data to a database that will be queried using SQL to detect discrepancies. Then,
you will use the Visual ETL tool from AWS Glue to check for any missing or duplicate data and upload the processed data
to the output folder.

Learning objectives
  - Create a Storage Area to Store the Input Files
  - Read the Raw Data to a Database
      - Create a Glue Crawler.
      - Configure the S3 input folder as the data source.
      - When prompted to create a new IAM role, add the suffix mlsc01 to the predefined role name so it's easy to find later.
      - Create and add a database.
      - Run the crawler and write the raw data to this database.
  - Run SQL Queries and Detect Data Discrepancies
      - Launch Amazon Athena and run SQL queries to detect null values against the age feature, check the number of observations
      that fall outside $250,000, and determine the format of first-name and last-name features.
  - Fix the Data Discrepancies
      - Use AWS Glue Visual ETL to configure an input S3 bucket to read the raw data, change the schema, assign proper data types,
        fill missing values against the age feature, filter data, and ignore rows whose salary is greater than $250,000.
      - Run a SQL query to convert the first and last names to lower case and remove the blank spaces in the fields.
      - Finally, write the formatted data to the output folder of the S3 bucket.

   ------------------------------------------------------


Solution

    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 Region.
    Download the input dataset Employee_lab.csv from the lab GitHub repo

Create a Storage Area to Store the Input Files

    From the console, navigate to S3.
    Click Create bucket and enter a globally unique bucket name.
    Scroll down and click Create bucket.
    Click on the newly created bucket. Choose Create folder, name the folder input and click on Create folder.
    Repeat the process and create another folder named output.
    From the list of folders, choose the input folder, click Upload, Add files, select the Employee_lab.csv file that you previously downloaded and click Open.
    Scroll down to the bottom of the console page and click Upload.
    Click the Close button to close the upload screen.

Read the Raw Data to a Database

    Search for Athena and select the service in the AWS console.
    Click on Launch query editor.
    Click Edit Settings to set up a query result location.
    Choose Browse S3 and click on the name of the bucket you previously created.
    Select the radio button for the output folder, click Choose, and Save.
    Choose Amazon Athena from the breadcrumbs at the top so that you are back on the Athena main page.
    Click Launch query editor again.
    From the Tables and views section, click to expand the Create dropdown menu.
    Select AWS Glue Crawler, give the crawler a name, and click Next.
    Click Add a data source.
    Under the S3 path section, click on Browse S3, and click on the name of the bucket you previously created.
    Select the radio button for the input folder, click Choose, and Save. NOTE: Ensure there is a forward slash at the end of the path (input/). If the slash is present but you still see an error message, click anywhere outside of the path input field and it should disappear.
    Keep all default options and click Add an S3 data source and Next.
    On the IAM role configuration settings page, click on Create new IAM role.
    Add the suffix mlsc01 to the predefined role name and click Create. Make sure the role is selected in the dropdown. If not, click the refresh button. Click Next.
    Click on Add database. Provide a name for the database and click Create database. NOTE: This action will complete in a new tab.
    Go back to the previous AWS Glue Set output and scheduling tab, refresh the Target database section, and select the new database you just created from the dropdown. Click Next.
    Scroll down and click Create crawler.
    Once the crawler is created successfully, click on Run Crawler.
    Scroll down, and under the Crawler runs tab, you will see the crawler in Running state turning into Completed state after successful completion.
    Expand the Data Catalog section from the left-hand menu.
    Under Databases click Tables to see a new table and the database you just created. If you don't see it, please click the refresh button.

Run SQL Queries and Detect Data Discrepancies

    From the Athena main page, click on Launch query editor. Refresh the Data section and make sure the database you created is selected and the new input table is listed under the Tables and Views section.

    Click the + next to the input table to see the data types of all the columns.

    Copy the following SQL query to the query window and click the Run button to execute. You should see the query output displaying 100 records in the table.

    select count(*) from input

    Next, follow the same process and run the following query. You will see there are two records where age is null.

    select count(*) from input where age is null

    Next, run the following query to display all the columns whose age is null.

    select * from input where age is null

    Now that we have detected records containing null values, let's check for outliers. Run the following query to check if employees make over $250,000. You will see two records that match this criteria.

    select * from input where salary > 250000

Fix the Data Discrepancies

    Search for AWS Glue and launch the service from the AWS console.

    Click on Set up roles and users. Click Choose roles, click the checkbox next to the role ending in mlsc01 that you created in a previous step. Click Confirm.

    Click Next. Accept the default values in the Grant Amazon S3 access screen. Click Next.

    Accept the default values in the Choose a default service role screen. Click Next.

    Review all the values and click Apply changes.

    From the main AWS Glue page, click Author and edit ETL jobs.

    Under the Create job section, click Visual ETL.

    From the Sources tab, click Amazon S3 to add that node to the visual panel.

    Click on the Amazon S3 node from within the panel to set its properties.

    Click Browse S3, click the name of the S3 bucket you created previously, and click the radio button next to the input folder. Click Choose.

    From the Data format dropdown, select CSV. The input data will be previewed under the Data preview section.

    At the top of the screen, give this job a name and click Save at the top right to save your work thus far.

    Click the blue + button in the panel view to add the next node.

    Switch to the Transforms tab, click the Change Schema node. If it didn't connect automatically, connect the output of the S3 node to the input of the Change Schema node.

    Under the Change Schema (Apply Mapping) section, change the data type of employee_id, age, and salary from string to int.

    Save the project.

    Click the blue + button to add the next node. Choose Fill missing values from the Transforms section. Click the resulting node in the panel. Under the Data field with missing values section, select age.

    You might see a data preview failure, which is a known issue with Glue version 4.0. If you don't see the error, AWS has addressed this issue, and you can skip to step 20. If you see the error, continue to the next step.

    Switch to the Job details tab. From the Glue version drop-down menu, select Glue version 2.0 and click Save in the top right.

    Click Run to rerun the job. Switch to the Runs tab to monitor the job. It may take close to 3 minutes for the job to complete successfully.

    After the run status shows Succeeded, switch back to the Visual tab.

    Scroll to the right in the Data preview section. You will see a new column named age_filled. As you scroll down, you will see that the null age values are no longer present in this new column.

    Click the blue + button to add the next node. From the Transforms tab, click Filter.

    Select that node, from the Transform panel on the right, click Add condition.

    From the Key dropdown menu, select salary.

    From the Operation dropdown, select the > operation. Enter 250000 in the Value textbox. The data preview will display the two outliers we saw when we ran the SQL query.

    From the Operation dropdown, select the < operation. The Data preview is updated to list all the values except the two outliers.

    Save the project.

    Click the blue + button to add the next node. From the Transforms section, choose SQL Query.

    Paste the following query under the SQL query section:

    select lower(replace(first_name, ' ','')) as first_name, lower(replace(last_name, ' ','')) as last_name, age_filled as age, gender, department, salary from myDataSource

    Ensure the Data preview section displays the column names as expected.

    Click the blue + button to add the final target node. Switch to the Targets tab, and click Amazon S3. If the nodes are not connected, connect the output of the SQL query node to the target S3 bucket node.

    Click the resulting node to configure it. From the Format dropdown, select CSV.

    From the drop-down under Compression Type select None.

    Under S3 Target Location section, click Browse S3, click the name of your bucket, select the radio button next to the output folder, and click Choose.

    Click Save to save the job and click Run to execute it.

    Switch to the Runs tab and ensure a new job has been triggered and is in the Running state. The job may take close to 4 minutes to complete.

    Once the job completes, navigate to S3 in the console and click the bucket you created previously. Choose the output folder, and click the file that starts with run and ends with 00000.

    Click the Download button at the top. Open the downloaded file to confirm that the data has been preprocessed.


   ------------------------------------------------------

   bucket name: data-prep-athena-glue-pa
   Athena Crawler: data-prep-crawler
   Athena DB: data-prep-db
   Glue Job Name:



