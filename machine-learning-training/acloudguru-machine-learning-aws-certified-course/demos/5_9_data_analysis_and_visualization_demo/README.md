------------------------------------------------------
5.9 Demo: Data Analysis and Visualization
------------------------------------------------------

  Resources:


  Note: Downloaded demo files to:
    C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\5_9_data_analysis_and_visualization_demo
     -> for Car Dataset (car_data.csv)
     -> my-manifest.json
     -> Box Plot Jupyter Notebook (box-plot-example.ipynb)

     jupyter notebook:
      https://jupyter.org/

  Use Case:
    You work for a company that owns several car dealerships.  You've been tasked with analyzing and visualizing
    some different data aspects about sales and possibly use it to predict future sales.
      - which month generates the most sales?
      - which salesman sold the most cars?
      - which dealership sold the most cars.
      - in what year was the average price of a Corvette greater than $100K?
      - which of those cars has several data points in the box plot's far right upper quartile?
     Final results
       - answer above business questions
       - visualization showing your results

    Use Quicksight or Jupyter Notebooks to analyze and visualize the data
      -> drag and drop method, then use Quicksight
      -> more hands on, then use Jupyter Notebooks

    Flow:

       S3 (csv)  ---> Amazon Quicksight (answer most questions)
                 ---> SageMaker / Jupyter Notebook for last question



  # first create car-analysis bucket and upload car_data.csv
  AWS Console -> S3 -> bucket name: car-analysis-data-lab -> Create Bucket
    -> Upload -> Add files -> car_data.csv -> upload


  AWS Console -> Quicksight
       -> Sign up for quicksight (note: 4 free users for 30 days, $250+/mo)
       -> after QS account is created

       -> New analysis -> New Dataset -> Data Source: S3, Data source name: car-data-source,
             Upload a manifest file: my-manifest.json -> Upload
              -> Connect

              -> Edit and Preview -> Examine data
                  -> Shows each colunm (car, year, engine_hp, ...) along with their data type (string, int, decimal, date, ...)
                  -> Could "add a calculated field", to combine two values or creates new fields,
                            add a filter which gets rid of some of these attributes (by unchecking fields).
                  -> Save & Visualize
                     -> (in top center you can name analysis, e.g,: Car Data Anaysis
                     -> left tab options: Visualize, filter (to exclude/include attributes),
                        Story (create stories that are interactive videos or interactive storyboards) ,
                        Parameters (create parameters that allow you to interact with your dashboard features)
                    -> Visualization
                        -> Visual Types (include bar charts, stacked bar charts, Pivot table, Pie Chart, ....)
                        -> 1st business questions: which month generates the most car sales
                           -> select "vertical bar chart" (since comparison question)
                              X-axis: sold month
                              value : car (count)
                                  -> now it shows the number of cars sold each month
                                  -> sold most cars in may (417)

                        -> 2nd business questions: which salesman sold the most car
                           -> Add -> Visual
                           -> select "pie chart" (since composition of data question)
                              Group/Color: saleman
                              value : car (count)
                                  -> now it shows the number of cars sold by each salesman
                                  -> Saleman 3 sold the most cars (1054)

                        -> 2nd-b business questions: which salesman sold the most car
                           -> Add -> Visual
                           -> select "vertical bar chart" (since composition of data question)
                              x-axis: saleman
                              Group/color for bars: car
                                  -> now it shows the number of each car type sold by each salesman
                                  -> Saleman 3 sold most cars by all 3 car types

                        -> 2nd-c business questions: which salesman sold the most car
                           -> Add -> Visual
                           -> select "Cluster bar combo chart" (since composition of data question)
                              x-axis: saleman
                              Group/color for bars: car
                              lines: car(count)
                                  -> now it shows the number of each car type sold by each salesman
                                  -> Saleman 3 sold most cars by all 3 car types

                        -> 3rd business questions: which dealership sold the most car (by car type)
                           -> Add -> Visual
                           -> select "Heat Map" (since composition of data question)
                              rows: dealership
                              columns: car
                              values (measure): car(count)
                                  -> right click -> Format visual -> Data Lables -> check: show data labels
                                  -> now heat map by dealership and by the 3 car types with labels showing number of cars
                                  -> Uptown car sold the most cars

                        -> 3rd-b business questions: which dealership sold the most car
                           -> Add -> Visual
                           -> select "Pie chart" (since composition of data question)
                              Group/Color: dealership
                              value (measure): car(count)
                                  -> Uptown car sold the most cars

                        -> 4th business questions: which year was average corvette sold price over $100k
                           -> Add -> Visual
                           -> select "Line chart" (since ??? question)
                              x-axis: sold (year)
                              value (measure): price(average)
                              color: car
                                  -> in 2005, the  average corvette price was ~$102k

         Quicksight Notes:
           - Quicksight stores all of the datasets in memory in a data store called SPICE.
             SPICE capactiy for region shown in upper left corner.
           - Data source for data set include upload file, S3, Athena, RDS, Redshif, Aurora, Spark, ...
           - Manifest file (JSON): info on what your data file's location (e.g. S3 bucket) and its format (e.g. CSV, ...)

            my-manifest.json contents:
            {
                "fileLocations": [
                    {
                        "URIPrefixes": [
                            "s3://<YOUR_BUCKET_NAME>/"
                        ]
                    }
                ],
                "globalUploadSettings": {
                    "format": "CSV",
                    "delimiter": ",",
                    "textqualifier": "'",
                    "containsHeader": "true"
                }
            }


  AWS Console -> SageMaker
     # missing from lab - creating a domain
     -> Getting Started -> Set up SageMaker Domain -> Set up for Single user (Quick setup)
     -> Add User: pat -> User profile: Name: pat-profile, Execution Role: AmazonSageMaker-ExecutionRole-20240518
        -> ...

    -> Open Studio   -> Notebook (left tab) -> Notebook Instances -> Create notebook instance
       Notebook instance name: my-notebook-inst -> Create an IAM role -> S3: Any S3 bucket -> Create role
       -> Create notebook instance
       -> Open Jupyter -> Files (tab) -> New -> (show all the different frameworks available) -> conda_python3
          -> (rename) car-data-box-plot-ex

       Note:
         SageMaker Examples (tab): lots of example Jupyter Notebooks for various ML datasets
         Conda (tab): can install additional Conda libraries

       car-analysis-data-lab jupyter notebook code:

         >>> # Importing the important libraries
         >>> import boto3
         >>> import pandas as pd
         >>> from sagemaker import get_execution_role
         >>> import numpy as np
         >>> import matplotlib.pyplot as plt
         >>> %matplotlib inline
         >>>
         >>> #Getting the car data from S3
         >>> role = get_execution_role()
         >>> bucket='car-analysis-data-lab'
         >>> data_key = 'car_data.csv'
         >>> data_location = 's3://{}/{}'.format(bucket, data_key)
         >>> print(data_location)
         >>>
         >>> # load data
         >>> df = pd.read_csv(data_location)
         >>> df.head()
         >>>
         >>> df_vet = df[df['car'] == 'Corvette']
         >>> df_mustang = df[df['car'] == 'Mustang']
         >>> df_camaro = df[df['car'] == 'Camaro']
         >>>
         >>> # create box-plot Engine HP for each car
         >>> data = [df_camaro['engine_hp'], df_vet['engine_hp'], df_mustang['engine_hp']]
         >>> plt.boxplot(data, vert=False)
         >>> plt.title('Engine HP Box Plot')
         >>> plt.xlabel('Engine HP')
         >>> plt.yticks([1, 2, 3], ['Camaro', 'Corvette', 'Mustang'])
         >>>
         >>> #Mustang HP shows 4 large HP outliers
         >>> plt.show()

  Clean up:

      -> File -> Close and halt (notebook) -> Logout
         my-notebook-inst -> Action -> Stop -> could also delete

