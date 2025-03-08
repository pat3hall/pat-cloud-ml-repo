------------------------------------------------------
2.10 Using Amazon S3 as a Machine Learning Lab


About this lab

Imagine you are a starting Data Engineer. You have been tasked with preparing an environment for model building.
In order to complete this task you need to ingest a csv file into S3 and then load that data source into a Jupyter
Notebook. Finally you need to save that data back into S3 under a different table.

Learning objectives
  - Prepare the Environment
  - Ingest Data Into SageMaker


    ------------------------------------------------------
Ingesting Data Into S3

    Go to the top of the page and in the search bar search for S3
    Select the S3 service to open the Amazon S3 page.
    Click Create Bucket
    Name the Bucket something unique
    Go to the bottom of page and click Create Bucket
    Once created, click on the Bucket name
    Go to https://open.toronto.ca/dataset/parking-tickets/
    Click Download Data
    Download parking-tickets-2022 and unzip on your computer
    Go back to your S3 Bucket page
    Click Upload
    Click Add Files and select Parking_Tags_Data_2022.000.csv from your unzipped parking-tickets-2022 file.
    Click Upload

Working with Data in Jupyter Notebooks

    Go to the top of the page and in the search bar search for SageMaker
    Click on Open Jupyter to the right of your notebook name
    On the right side of the Jupyter page, click New then select the dropdown option conda_python3
    In the first cell enter

  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt

    Click Shift and Enter to run the cell
    In the next cell enter

   df = pd.read_csv('s3://YOURBUCKETNAME/Parking_Tags_Data_2022.000.csv')
   df.head()

    Replace YOURBUCKETNAME with the name of your bucket
    Click Shift and Enter to run the cell
    In that same cell add an 11 inside the parentheses

   df.head(11)

    Click Shift and Enter to run the cell
    In a new cell enter

   df.drop(columns='location3')

    Click Shift and Enter to run the cell
    In a new cell enter

   df.to_csv('s3://YOURBUCKETNAME/Result.csv')

    Replace YOURBUCKETNAME with the name of your bucket
    Click Shift and Enter to run the cell
    Go back to the S3 bucket tab and click on the bucket.
    Verify that Result.csv is a file in the bucket.

Congratulations, you have completed this hands-on lab!

    ------------------------------------------------------
    code: lab 2.10 read S3 csv, drop column, write results csv to S3

        >>> import numpy as np
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt

        >>> df = pd.read_csv('s3://my-ml-repo-bkt/Parking_Tags_Data_2022.000.csv')
        >>> df.head(11)

        >>> df.drop(columns='location3')

        >>> df.to_csv('s3://my-ml-repo-bkt/Result.csv')

    ------------------------------------------------------



