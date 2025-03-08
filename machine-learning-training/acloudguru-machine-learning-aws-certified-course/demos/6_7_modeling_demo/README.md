------------------------------------------------------
6.7 Demo: Modeling

  Resouces:

    Note: Downloaded demo files to:
      C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\6_7_demo_modeling

    UFO Sightings Dataset (uso_fullset.csv)
      https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter6/ufo_fullset.csv

    Jupyter Notebook (ufo-modeling-lab.ipynb)
      https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter6/ufo-modeling-lab.ipynb

    K-Kmeans SageMaker Documentation (no longer at provided location - now at below location)
      https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html

    K-Kmeans Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/k-means-api-config.html


    SageMaker trained model Deserialization
      https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
      SageMaker Output model
        - SageMaker models are stored as 'model.tar.gz" in the S3 bucket specified in 'OutputDataConfig' S3OutputPath parameter
          of the create_training_job call.
       -  When model.tar.gz is untarred, it contains model_algo-1, which is a serialized Apache MXNet object.
       - use the following to load the k-means model into memory and view it
         import mxnet as mx
         print(mx.ndarray.load('model_algo-1'))


  From ORiely Hands ON Machine Learning (chap 9):

    The K-Means Algorithm
      The K-Means algorithm is one of the fastest clustering algorithms, and also one of the simplest:
        - First initialize k-centroids randomly: e.g., k-distinct instances are chosen randomly from the dataset and
          the centroids are placed at their locations.
        - Repeat until convergence (i.e., until the centroids stop moving):
           - Assign each instance to the closest centroid.
           - Update the centroids to be the mean of the instances that are assigned to them.


  Identify Sensor Locations
    - deploy a network of 10 sensors across the globe
    - locate these sensors in the center of the 10 most likely locations for UFO sightings
    - What type of generalization are trying to make?
    - Do we have enough data? what does our data look like?
    - What algoritm can help use solve this problem?
    - Where should we launch these sensors?

  Dataset attributes:
      "reportedTimestamp","eventDate","eventTime","shape","duration" (how many seconds),"witnesses","weather",
      "firstName","lastName", "latitude","longitude","sighting" (always 'yes'),"physicalEvidence" (Y=Yes, N=No),
      "contact" (Y=Yes, N=No),"researchOutcome ("explained", "unexplained", "probable")

  Final Results:
    - 10 locations (latitude and longitude
    - Map of locations

  SageMaker and K-means
    - use SageMaker and the K-means clustering algorithm to locate teh 10 best locations
    - Visualize teh locations in QuickSight
    - K-means is an unsupervised learning algorithm that attempts to find discrete groupings within data
      - use latitude and longitude as the values we want to group together (find similarity in)
      - set K-Means hyperparameter k = 10 since we want to find 10 locations


  Steps:
    - create S3 bucket and upload ufo data
    - In SageMaker:
       - Create Jupyter Notebook and prepare the data
       - train model using K-means
       - inference from model the 10 locations
    - In QuickSight:
       - visualize the data

  # first create modeling-ufo-lab bucket and upload ufo_fulset.csv and ufo-modeling-lab.ipynb
  AWS Console -> S3 -> bucket name: modeling-ufo-lab1 -> Create Bucket
    -> Create folder -> ufo_dataset -> create
    -> ufo_dataset folder -> Upload -> Add files -> ufo_fulset.csv -> upload
    -> Upload -> Add files -> ufo-modeling-lab.ipynb -> upload


  AWS Console -> SageMaker
    # actually, I reused last lab instance
    -> Notebook (left tab) -> Notebook Instances -> Create notebook instance
       Notebook instance name: my-notebook-inst,  instance type: ml.t3.medium, platform: AL2, Jupyter Lab 3
          -> Create an IAM role -> S3: Any S3 bucket -> Create role
       -> Create notebook instance
       -> Open Jupyter -> Upload -> ufo-modeling-lab.ipynb -> select


  Code: UFO modeling lab code

         >>> # Import libraries
         >>> import pandas as pd
         >>> import numpy as np
         >>> from datetime import datetime

         >>> import boto3
         >>> from sagemaker import get_execution_role
         >>> import sagemaker.amazon.common as smac

         >>> # Step 1: Loading the data from Amazon S3
         >>> role = get_execution_role()
         >>> bucket = 'modeling-ufo-lab1'
         >>> prefix = 'ufo_dataset'
         >>> data_key = 'ufo_fullset.csv'
         >>> data_location = 's3://{}/{}/{}'.format(bucket, prefix, data_key)

         >>> df = pd.read_csv(data_location, low_memory=False)
         >>> df.head()

         >>> df.shape()
             (18000, 15)

         >>> # Step 2: Cleaning, transforming, and preparing the data
         >>> # Create another DataFrame with just the latitude and longitude attributes
         >>> df_geo = df[['latitude', 'longitude']]
         >>> df_geo.head()
         >>> df_geo.info()
             <class 'pandas.core.frame.DataFrame'>
             RangeIndex: 18000 entries, 0 to 17999
             Data columns (total 2 columns):
              #   Column     Non-Null Count  Dtype
             ---  ------     --------------  -----
              0   latitude   18000 non-null  float64
              1   longitude  18000 non-null  float64
             dtypes: float64(2)
             memory usage: 281.4 KB

         >>> # check for missing values
         >>> missing_values = df_geo.isnull().values.any()
         >>> print('Are there any missing values? {}'.format(missing_values))
         >>> if(missing_values):
         >>>     df_geo[df_geo.isnull().any(axis=1)]

             Are there any missing values? False

         >>> # Next, let's go ahead and transform the pandas DataFrame (our dataset) into a numpy.ndarray.
         >>> # When we do this each row is converted to a Record object.
         >>> # According to the documentation, this is what the K-Means algorithm expects as training data.
         >>> # This is what we will use as training data for our model.

         >>> # dataframe.values.astype('float32') returns numpy.ndarray of float32 objects
         >>> data_train = df_geo.values.astype('float32')
         >>> data_train
             array([[  47.329445, -122.57889 ],
                   [  52.664913,   -1.034894],
                   [  38.951668,  -92.333885],
                   ...,
                   [  36.86639 ,  -83.888885],
                   [  35.385834,  -94.39833 ],
                   [  29.883055,  -97.94111 ]], dtype=float32)

         >>> # Step 3: Create and train our model
         >>> # In this step we will import and use the built-in SageMaker K-Means algorithm. We will set the number of cluster to 10 (for
         >>> # our 10 sensors), specify the instance type we want to train on, and the location of where we want our model artifact to live.
         >>>
         >>> # See the documentation for input training
         >>> #   https://docs.aws.amazon.com/sagemaker/latest/dg/k-means-api-config.html

         >>> from sagemaker import KMeans
         >>>
         >>> num_clusters = 10
         >>> output_location = 's3://' + bucket + '/model-artifacts'
         >>>
         >>> kmeans = KMeans(role=role,
         >>>                instance_count=1,
         >>>                instance_type='ml.c4.xlarge',
         >>>                output_path=output_location,
         >>>                k=num_clusters)


         >>> job_name = 'kmeans-geo-job-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
         >>> print('Here is the job name {}'.format(job_name))
             Here is the job name kmeans-geo-job-20240624172337

         >>> # use kmeans.record_set() to make sure training data is in correct format
         >>> %%time
         >>> kmeans.fit(kmeans.record_set(data_train), job_name=job_name)

             INFO:sagemaker.image_uris:Same images used for training and inference. Defaulting to image scope: inference.
             INFO:sagemaker.image_uris:Ignoring unnecessary instance type: None.
             INFO:sagemaker:Creating training-job with name: kmeans-geo-job-20240620221726
             2024-06-20 22:17:31 Starting - Starting the training job...
             2024-06-20 22:17:48 Starting - Preparing the instances for training...
             2024-06-20 22:18:15 Downloading - Downloading input data...
             2024-06-20 22:18:55 Downloading - Downloading the training image.......
             . . .
             2024-06-20 22:20:39 Completed - Training job completed
             Training seconds: 144
             Billable seconds: 144
             CPU times: user 708 ms, sys: 44.7 ms, total: 753 ms
             Wall time: 3min 43s

             # SageMaker store output at:
             # Amazon S3 -> modeling-ufo-lab1 model-artifacts/kmeans-geo-job-20240624172337/output/model.tar.gz


         >>> # Step 4: Viewing the results
         >>> # In this step we will take a look at the model artifact SageMaker created for us and stored onto S3. We have to do a few special things
         >>> # to see the latitude and longitude for our 10 clusters (and the center points of those clusters)
         >>>
         >>> # See the documentation of deserilization here
         >>> #     https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
         >>>
         >>> # At this point we need to "deserilize" the model artifact. Here we are going to open and review them in our notebook instance.
         >>> # We can unzip the model artifact which will contain model_algo-1. This is just a serialized Apache MXNet object. From here we
         >>> # can load that serialized object into a numpy.ndarray and then extract the clustered centroids from the numpy.ndarray.
         >>>
         >>> # After we extract the results into a DataFrame of latitudes and longitudes, we can create a CSV with that data, load it onto S3 and
         >>> # then visualize it with QuickSight.

         >>> import os
         >>> model_key = 'model-artifacts/' + job_name + '/output/model.tar.gz'

         >>> # download model output from S3 to notebook instance, then unzip to extract "model_algo-1" output file
         >>> boto3.resource('s3').Bucket(bucket).download_file(model_key, 'model.tar.gz')
         >>> os.system('tar -zxvf model.tar.gz')
         >>> os.system('unzip model_algo-1')

         >>> !pip install mxnet

         >>> # convert model output to ndarray
         >>> import mxnet as mx
         >>> Kmeans_model_params = mx.ndarray.load('model_algo-1')


         >>> # convert model output ndarray to panda DF and add column names
         >>> cluster_centroids_kmeans = pd.DataFrame(Kmeans_model_params[0].asnumpy())
         >>> cluster_centroids_kmeans.columns=df_geo.columns
         >>> cluster_centroids_kmeans

              	latitude 	longitude
             0 	35.336853 	-98.741165
             1 	49.873829 	-3.797668
             2 	-4.999058 	112.205666
             3 	34.935562 	-118.706398
             4 	31.744030 	-82.604263
             5 	41.340214 	-75.298149
             6 	46.147224 	-119.763893
             7 	62.170712 	-148.799774
             8 	40.933262 	-87.650185
             9 	1.034187 	-67.699471

         >>> # Let's go ahead and upload this dataset onto S3 and view within QuickSight

         >>> # Note: StringIO model is an in-memory file-lib object. This object can be used as input or output to the most function
         >>> #   that would expect a standard file object
         >>> from io import StringIO

         >>> # store CSV results data on S3 in bucket at: 'results/ten_locations_kmeans.csv
         >>> csv_buffer = StringIO()
         >>> cluster_centroids_kmeans.to_csv(csv_buffer, index=False)
         >>> s3_resource = boto3.resource('s3')
         >>> s3_resource.Object(bucket, 'results/ten_locations_kmeans.csv').put(Body=csv_buffer.getvalue())


AWS console -> QuickSight -> create account (again)

   QuickSight -> Point Dataset to S3 results file -> Points on Map (visualization) -> Geospatial Data -> add longitude & latitude


