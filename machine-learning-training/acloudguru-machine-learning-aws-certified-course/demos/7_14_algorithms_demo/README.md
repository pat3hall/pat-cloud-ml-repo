------------------------------------------------------
7.14 Demo: Algorithms


  Resources

    Note: Downloaded github demo files to:
     C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\7_14_algorithms_demo

     -> Jupyter Notebook:
        ufo-algorithms-lab.ipynb
     -> dataset data:
          ufo_fullset.csv

  Identifying UFO Sightings Legitimacy
    - build model that can predict the legitimacy of a UFO sighting based on the information supplied by
      the submitter
    - can use the ground truth dataset used in the previous lab on whether sightings could be explained
      (as a hoax or other natural explanations), unexplained, or probable

  To do:
    - build a model to determine whether newly reported UFO sightings are legitimate or not (explained,
      unexplained, or probable)
    - require the model be at least 90% accurate
    - Which algorithm would you choose?
    - What does our data look like? Do we need to transform and prepare our data? Are we missing an values?
    - How accurate is our model?

  Final Results (for each algorithm):
    - Model artifact store in S3
    - Model validation metrics (accuracy, recall, precision, and the F1 score)
    - model error rate

   Additional Suggested Steps:
     - build a model usingn the XGBoost algorithm as a multi-classification problem with 'researchOutcomes' as
       the target attribute
         - choosing XGBoost because: simple to implement, and it only needs two hyperparameters with 35 optional
           hyperparameters as well.  It's really simple to implement
         - XGBoost can be used for different types of problems, like ranking problems or regression problems,
         - we're going to use it as a multi-classification problem.
         - since we're not exactly sure which attributes are most important, then we need to use a classification
           algorithm over a clustering algorithm
     - Goal is to minimize the training and validation error

     - also build a model usingn the Linear Learner algorithm as a multi-classification problem with 'researchOutcomes'
       as the target attribute
         - train different models and determine which attributes from our data are the most determining factors.
         - Linear Learner has built-in hyperparameter tuning
     - Goal is to maximize the training accuracy (and other metrics like precision, F1 score, and recall)


     Flow:
         S3 (dataset)  ---> SageMaker  --> Notebook  --> train  --> create model --> upload to S3

         Convert data to numeric format
           - XGBoost and Linear Learner both expect dataset to be in numeric format
         Split dataset to training, validation, and testing sets

         Train via algorithm

         logs
            - in cloudwatch and in Notebook after training job completes


  # first create modeling-ufo-lab bucket and upload ufo_fulset.csv  - performed in previous lab
  AWS Console -> S3 -> bucket name: modeling-ufo-lab1 -> Create Bucket
    -> Create folder -> ufo_dataset -> create
    -> ufo_dataset folder -> Upload -> Add files -> ufo_fulset.csv -> upload


  AWS Console -> SageMaker
    # actually, I reused last lab instance
    -> Notebook (left tab) -> Notebook Instances -> Create notebook instance
       Notebook instance name: my-notebook-inst,  instance type: ml.t3.medium, platform: AL2, Jupyter Lab 3
          -> Create an IAM role -> S3: Any S3 bucket -> Create role
       -> Create notebook instance
       -> Open Jupyter -> Upload -> ufo-algorithms-lab.ipynb -> select
          -> (rename) car-data-box-plot-ex

  Code: UFO algorithm lab code

         >>> # First let's go ahead and import all the needed libraries.
         >>> import pandas as pd
         >>> import numpy as np
         >>> from datetime import datetime
         >>> import io
         >>> import sagemaker.amazon.common as smac
         >>>
         >>> import boto3
         >>> from sagemaker import get_execution_role
         >>> import sagemaker
         >>>
         >>> import matplotlib.pyplot as plt
         >>> import seabomodelingrn as sns

         >>> # Let's get the UFO sightings data that is stored in S3 and load it into memory.
         >>> role = get_execution_role()
         >>> bucket='modeling-ufo-lab1'
         >>> sub_folder = 'ufo_dataset'
         >>> data_key = 'ufo_fullset.csv'
         >>> data_location = 's3://{}/{}/{}'.format(bucket, sub_folder, data_key)
         >>>
         >>> df = pd.read_csv(data_location, low_memory=False)
         >>> df.head()
         >>>
             	reportedTimestamp	eventDate	eventTime	shape	duration	witnesses	weather	firstName	lastName	latitude	longitude	sighting	physicalEvidence	contact	researchOutcome
             0	1977-04-04T04:02:23.340Z	1977-03-31	23:46	circle	4	1	rain	Ila	Bashirian	47.329444	-122.578889	Y	N	N	explained
             1	1982-11-22T02:06:32.019Z	1982-11-15	22:04	disk	4	1	partly cloudy	Eriberto	Runolfsson	52.664913	-1.034894	Y	Y	N	explained
             . . .

         >>> # Step 2: Cleaning, transforming, analyize, and preparing the dataset
         >>>
         >>> # Let's check to see if there are any missing values
         >>> missing_values = df.isnull().values.any()
         >>> if(missing_values):
         >>>     display(df[df.isnull().any(axis=1)])
         >>>
         >>> # Note: Found just 2 missing values for 'shape' in row 1024 and 2048
         >>>
         >>> # determine most common shape, and use it to replace missing values
         >>> df['shape'].value_counts()

             shape
             circle      6047
             disk        5920
             light       1699
             . . .

         >>> # Replace the missing values with the most common shape
         >>> #    Note: pandas.DataFrame.fillna()    Fills NA/NaN values using the specified method
         >>> df['shape'] = df['shape'].fillna(df['shape'].value_counts().index[0])


         >>> # Let's go ahead and start preparing our dataset by transforming some of the values into the correct data types.
         >>> # Here is what we are going to take care of.
         >>>
         >>> #  1. Convert the reportedTimestamp and eventDate to a pandas datetime data types.
         >>> #  2. Convert the shape and weather to a pandas category data type.
         >>> #  3. Map the physicalEvidence and contact from 'Y', 'N' to 0, 1.
         >>> #  4. Convert the researchOutcome to a pandas category data type (target attribute).


         >>> df['reportedTimestamp'] = pd.to_datetime(df['reportedTimestamp'])
         >>> df['eventDate'] = pd.to_datetime(df['eventDate'])
         >>>
         >>> df['shape'] = df['shape'].astype('category')
         >>> df['weather'] = df['weather'].astype('category')
         >>>
         >>> df['physicalEvidence'] = df['physicalEvidence'].replace({'Y': 1, 'N': 0})
         >>> df['contact'] = df['contact'].replace({'Y': 1, 'N': 0})
         >>>
         >>> df['researchOutcome'] = df['researchOutcome'].astype('category')

         >>> df.dtypes

         >>> # Let's visualize some of the data to see if we can find out any important information.
         >>>
         >>> %matplotlib inline
         >>> sns.set_context("paper", font_scale=1.4)
         >>>
         >>> m_cts = (df['contact'].value_counts())
         >>> m_ctsx = m_cts.index
         >>> m_ctsy = m_cts.to_numpy()
         >>> f, ax = plt.subplots(figsize=(5,5))
         >>>
         >>> sns.barplot(x=m_ctsx, y=m_ctsy)
         >>> ax.set_title('UFO Sightings and Contact')
         >>> ax.set_xlabel('Was contact made?')
         >>> ax.set_ylabel('Number of Sightings')
         >>> ax.set_xticklabels(['No', 'Yes'])
         >>> plt.xticks(rotation=45)
         >>> plt.show()

             # Seaborn barplot() displayed visualizing 'was contacted made?' yes or no (no contact made in most sightings)
             #  contact No:    16600  Yes:   1400

         >>> m_cts = (df['physicalEvidence'].value_counts())
         >>> m_ctsx = m_cts.index
         >>> m_ctsy = m_cts.to_numpy()
         >>> f, ax = plt.subplots(figsize=(5,5))
         >>>
         >>> sns.barplot(x=m_ctsx, y=m_ctsy)
         >>> ax.set_title('UFO Sightings and Physical Evidence')
         >>> ax.set_xlabel('Was there physical evidence?')
         >>> ax.set_ylabel('Number of Sightings')
         >>> ax.set_xticklabels(['No', 'Yes'])
         >>> plt.xticks(rotation=45)
         >>> plt.show()

             # Seaborn barplot() displayed visualizing 'was there physical evidence?' yes or no (no physical evidence in most sightings)
             #   physicalEvidence  No:    15313  Yes:     2687


         >>> m_cts = (df['shape'].value_counts())
         >>> m_ctsx = m_cts.index
         >>> m_ctsy = m_cts.to_numpy()
         >>> f, ax = plt.subplots(figsize=(9,5))
         >>>
         >>> sns.barplot(x=m_ctsx, y=m_ctsy)
         >>> ax.set_title('UFO Sightings by Shape')
         >>> ax.set_xlabel('UFO Shape')
         >>> ax.set_ylabel('Number of Sightings')
         >>> plt.xticks(rotation=45)
         >>> plt.show()

             # Seaborn barplot() displayed visualizing 'UFO Shapes?'
             #  shape: circle: 6049, disk: 5920, light: 1699, square: 1662, triangle: 1062, sphere: 1020, box: 200, oval: 199, pyramid: 189

         >>> m_cts = (df['weather'].value_counts())
         >>> m_ctsx = m_cts.index
         >>> m_ctsy = m_cts.to_numpy()
         >>> f, ax = plt.subplots(figsize=(5,5))
         >>>
         >>> sns.barplot(x=m_ctsx, y=m_ctsy)
         >>> ax.set_title('UFO Sightings by Weather')
         >>> ax.set_xlabel('Weather')
         >>> ax.set_ylabel('Number of Sightings')
         >>> plt.xticks(rotation=45)
         >>> plt.show()

             # Seaborn barplot() displayed visualizing 'UFO Sightings by Weather'
             # weather: clear: 3206, mostly_cloudy: 3079, partly_cloudy: 2704, rain: 2605, stormy: 2162, fog: 2123, snow: 2121

         >>> m_cts = (df['researchOutcome'].value_counts())
         >>> m_ctsx = m_cts.index
         >>> m_ctsy = m_cts.to_numpy()
         >>> f, ax = plt.subplots(figsize=(5,5))
         >>>
         >>> sns.barplot(x=m_ctsx, y=m_ctsy)
         >>> ax.set_title('UFO Sightings and Research Outcome')
         >>> ax.set_xlabel('Research Outcome')
         >>> ax.set_ylabel('Number of Sightings')
         >>> plt.xticks(rotation=45)
         >>> plt.show()

             # Seaborn barplot() displayed visualizing 'UFO Sightings and Research Outcome'
             # researchOutcome explained: 12822, unexplained: 3308, probable: 1870

         >>> ufo_yr = df['eventDate'].dt.year  # series with the year exclusively
         >>>
         >>> ## Set axes ##
         >>> years_data = ufo_yr.value_counts()
         >>> years_index = years_data.index  # x ticks
         >>> years_values = years_data.to_numpy()
         >>>
         >>> ## Create Bar Plot ##
         >>> plt.figure(figsize=(15,8))
         >>> plt.xticks(rotation = 60)
         >>> plt.title('UFO Sightings by Year')
         >>> plt.ylabel('Number of Sightings')
         >>> plt.xlabel('Year')
         >>>
         >>> years_plot = sns.barplot(x=years_index[:60],y=years_values[:60])

             # Seaborn barplot() displayed 'UFO Sightings By Year'
             # shows fairly even distribution of sightings for each year

         >>> # display pandas correlation between attributes (needed to add 'numeric_only=True':
         >>> df.corr(numeric_only=True)
             .  .  .
             	                duration	witnesses	latitude	longitude	physicalEvidence	contact
             duration	        1.000000	0.020679	0.000243	-0.010529	0.016430	        0.015188
             witnesses	        0.020679	1.000000	0.010229	0.003449	0.009186	        -0.000651
             latitude	        0.000243	0.010229	1.000000	-0.394536	0.006465	        0.004284
             longitude	        -0.010529	0.003449	-0.394536	1.000000	-0.004519	        -0.004828
             physicalEvidence	0.016430	0.009186	0.006465	-0.004519	1.000000	        0.693276
             contact	        0.015188	-0.000651	0.004284	-0.004828	0.693276	        1.000000

             # df.corr() in video found:  some strong correlations with physical evidence and contact.
             # whenever there is contact, that there is physical evidence at least 69% of the time.

         >>> # Let's drop the columns that are not important.
         >>> #
         >>> # 1. We can drop sighting becuase it is always 'Y' or Yes.
         >>> # 2. Let's drop the firstName and lastName becuase they are not important in determining the researchOutcome.
         >>> # 3. Let's drop the reportedTimestamp because when the sighting was reporting isn't going to help us determine the
         >>> #    legitimacy of the sighting.
         >>> # 4. We would need to create some sort of buckets for the eventDate and eventTime, like seasons for example, but since
         >>> #    the distribution of dates is pretty even, let's go ahead and drop them.
         >>>
         >>> df.drop(columns=['firstName', 'lastName', 'sighting', 'reportedTimestamp', 'eventDate', 'eventTime'], inplace=True)
         >>> df.head()

             	shape	duration	witnesses	weather	latitude	longitude	physicalEvidence	contact	researchOutcome
             0	circle	4	1	rain	47.329444	-122.578889	0	0	explained
             1	disk	4	1	partly cloudy	52.664913	-1.034894	1	0	explained
             . . .

         >>> # Let's apply one-hot encoding
         >>> #
         >>> # We need to one-hot both the weather attribute and the shape attribute.
         >>> # We also need to transform or map the researchOutcome (target) attribute into numeric values. This is what the alogrithm
         >>> #   is expecting. We can do this by mapping unexplained, explained, and probable to 0, 1, 2.


         >>> # Let's one-hot the weather and shape attribute
         >>> #  Note: pandas.get_dummies() Convert categorical variable into dummy/indicator variables. (converts to one-hot values)
         >>> #  Note: FIXED! needed to add 'dtype='int' because get_dummies dtype default is now 'bool'
         >>>
         >>> df = pd.get_dummies(df, columns=['weather', 'shape'], dtype='int')
         >>>
         >>> # Let's replace the researchOutcome values with 0, 1, 2 for Unexplained, Explained, and Probable
         >>> df['researchOutcome'] = df['researchOutcome'].replace({'unexplained': 0, 'explained': 1, 'probable': 2})


         >>> # Let's randomize and split the data into training, validation, and testing.
         >>>
         >>> # 1. First we need to randomize the data.
         >>> # 2. Next Let's use 80% of the dataset for our training set.
         >>> # 3. Then use 10% for validation during training.
         >>> # 4. Finally we will use 10% for testing our model after it is deployed.



         >>> # Let's go ahead and randomize our data.
         >>> # pandas sample() returns are random sample of the size of 'frac' (fraction), since '1' returns full randomized dataframe
         >>> #   and "reset_index(drop=True)" drop previous dataframe index values
         >>> df = df.sample(frac=1).reset_index(drop=True)
         >>>
         >>> # Next, Let's split the data into a training, validation, and testing.
         >>> rand_split = np.random.rand(len(df))
         >>> train_list = rand_split < 0.8                       # 80% for training
         >>> val_list = (rand_split >= 0.8) & (rand_split < 0.9) # 10% for validation
         >>> test_list = rand_split >= 0.9                       # 10% for testing
         >>>
         >>>  # This dataset will be used to train the model.
         >>> data_train = df[train_list]
         >>>
         >>> # This dataset will be used to validate the model.
         >>> data_val = df[val_list]
         >>>
         >>> # This dataset will be used to test the model.
         >>> data_test = df[test_list]


         >>> # Next, let's go ahead and rearrange our attributes so the first attribute is our target attribute researchOutcome.
         >>> #  This is what AWS requires and the XGBoost algorithms expects. You can read all about it here in the documentation.
         >>> #      https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html#InputOutput-XGBoost
         >>>
         >>> #  After that we will go ahead and create those files on our Notebook instance (stored as CSV) and then upload them to S3.

         >>> # Simply moves the researchOutcome attribute to the first position before creating CSV files
         >>> pd.concat([data_train['researchOutcome'], data_train.drop(['researchOutcome'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
         >>> pd.concat([data_val['researchOutcome'], data_val.drop(['researchOutcome'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)

         >>> # Next we can take the files we just stored onto our Notebook instance and upload them to S3.
         >>> boto3.Session().resource('s3').Bucket(bucket).Object('algorithms_lab/xgboost_train/train.csv').upload_file('train.csv')
         >>> boto3.Session().resource('s3').Bucket(bucket).Object('algorithms_lab/xgboost_validation/validation.csv').upload_file('validation.csv')

         >>> # Step 3: Creating and training our model (XGBoost)
         >>> # This is where the magic happens. We will get the ECR container hosted in ECR for the XGBoost algorithm.

         >>> from sagemaker import image_uris
         >>> container = image_uris.retrieve('xgboost', boto3.Session().region_name, '1')

         >>> # Next, because we're training with the CSV file format, we'll create inputs that our training function can use as a pointer
         >>> #  to the files in S3, which also specify that the content type is CSV.
         >>>
         >>> s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/algorithms_lab/xgboost_train'.format(bucket), content_type='csv')
         >>> s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://{}/algorithms_lab/xgboost_validation'.format(bucket), content_type='csv')


         >>> # Next we start building out our model by using the SageMaker Python SDK and passing in everything that is required to
         >>> #   create a XGBoost model.
         >>> #
         >>> # First I like to always create a specific job name.
         >>> #
         >>> # Next, we'll need to specify training parameters.
         >>> #
         >>> # 1. The xgboost algorithm container
         >>> # 2. The IAM role to use
         >>> # 3. Training instance type and count
         >>> # 4. S3 location for output data/model artifact
         >>> # 5. XGBoost Hyperparameters
         >>> #     https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html

         >>>
         >>> # Finally, after everything is included and ready, then we can call the .fit() function which specifies the S3 location
         >>> #  for training and validation data.

         >>> # Create a training job name
         >>> job_name = 'ufo-xgboost-job-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
         >>> print('Here is the job name {}'.format(job_name))
         >>>
         >>> # Here is where the model artifact will be stored
         >>> output_location = 's3://{}/algorithms_lab/xgboost_output'.format(bucket)

             Here is the job name ufo-xgboost-job-20240701184208

         >>> sess = sagemaker.Session()
         >>>
         >>> xgb = sagemaker.estimator.Estimator(container, role, instance_count=1, instance_type='ml.m4.xlarge',
         >>>                                     output_path=output_location, sagemaker_session=sess)
         >>>
         >>> xgb.set_hyperparameters(objective='multi:softmax', num_class=3, num_round=100)
         >>>
         >>> data_channels = { 'train': s3_input_train, 'validation': s3_input_validation }
         >>> xgb.fit(data_channels, job_name=job_name)

             INFO:sagemaker:Creating training-job with name: ufo-xgboost-job-20240701191325
             2024-07-01 19:13:26 Starting - Starting the training job...
             . . .
             [98]#011train-merror:0.00893#011validation-merror:0.056299
             [19:16:07] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 68 extra nodes, 0 pruned nodes, max_depth=6
             [19:16:07] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 48 extra nodes, 0 pruned nodes, max_depth=6
             [19:16:07] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 64 extra nodes, 0 pruned nodes, max_depth=6
             [99]#011train-merror:0.008581#011validation-merror:0.055741

             2024-07-01 19:16:25 Uploading - Uploading generated training model
             2024-07-01 19:16:25 Completed - Training job completed
             Training seconds: 119
             Billable seconds: 119

         >>> # note: the training accuracy was (1 - 0.0085)*100 = 99.9925, but the validation accuracy was (1 - 0.0557)*100 = 95.53
         >>> #         so we may an unfitting issue
         >>>
         >>> print('Here is the location of the trained XGBoost model: {}/{}/output/model.tar.gz'.format(output_location, job_name))

             Here is the location of the trained XGBoost model: s3://modeling-ufo-lab1/algorithms_lab/xgboost_output/ufo-xgboost-job-20240701191325/output/model.tar.gz

         >>> # After we train our model we can see the default evaluation metric in the logs. The merror is used in multiclass
         >>> #   classification error rate. It is calculated as #(wrong cases)/#(all cases). We want this to be minimized (so we
         >>> #   want this to be super small).
         >>>
         >>>
         >>> # Step 4: Creating and training our model (Linear Learner)
         >>> #
         >>> #  Let's evaluate the Linear Learner algorithm as well. Let's go ahead and randomize the data again and get it ready
         >>> #  for the Linear Leaner algorithm. We will also rearrange the columns so it is ready for the algorithm (it expects
         >>> #  the first column to be the target attribute)

         >>> np.random.seed(0)
         >>> rand_split = np.random.rand(len(df))
         >>> train_list = rand_split < 0.8
         >>> val_list = (rand_split >= 0.8) & (rand_split < 0.9)
         >>> test_list = rand_split >= 0.9
         >>>
         >>>  # This dataset will be used to train the model.
         >>> data_train = df[train_list]
         >>>
         >>> # This dataset will be used to validate the model.
         >>> data_val = df[val_list]
         >>>
         >>> # This dataset will be used to test the model.
         >>> data_test = df[test_list]
         >>>
         >>> # This rearranges the columns
         >>> cols = list(data_train)
         >>> cols.insert(0, cols.pop(cols.index('researchOutcome')))
         >>> data_train = data_train[cols]
         >>>
         >>> cols = list(data_val)
         >>> cols.insert(0, cols.pop(cols.index('researchOutcome')))
         >>> data_val = data_val[cols]
         >>>
         >>> cols = list(data_test)
         >>> cols.insert(0, cols.pop(cols.index('researchOutcome')))
         >>> data_test = data_test[cols]
         >>>
         >>> # Breaks the datasets into attribute numpy.ndarray and the same for target attribute.
         >>> train_X = data_train.drop(columns='researchOutcome').values
         >>> train_y = data_train['researchOutcome'].values
         >>>
         >>> val_X = data_val.drop(columns='researchOutcome').values
         >>> val_y = data_val['researchOutcome'].values
         >>>
         >>> test_X = data_test.drop(columns='researchOutcome').values
         >>> test_y = data_test['researchOutcome'].values

         >>> #  Next, Let's create recordIO file for the training data and upload it to S3.

         >>> train_file = 'ufo_sightings_train_recordIO_protobuf.data'
         >>>
         >>> f = io.BytesIO()
         >>> smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
         >>> f.seek(0)
         >>>
         >>> boto3.Session().resource('s3').Bucket(bucket).Object('algorithms_lab/linearlearner_train/{}'.format(train_file)).upload_fileobj(f)
         >>> training_recordIO_protobuf_location = 's3://{}/algorithms_lab/linearlearner_train/{}'.format(bucket, train_file)
         >>> print('The Pipe mode recordIO protobuf training data: {}'.format(training_recordIO_protobuf_location))

             The Pipe mode recordIO protobuf training data: s3://modeling-ufo-lab1/algorithms_lab/linearlearner_train/ufo_sightings_train_recordIO_protobuf.data


         >>> #  Let's create recordIO file for the validation data and upload it to S3

         >>> validation_file = 'ufo_sightings_validatioin_recordIO_protobuf.data'
         >>>
         >>> f = io.BytesIO()
         >>> smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
         >>> f.seek(0)
         >>>
         >>> boto3.Session().resource('s3').Bucket(bucket).Object('algorithms_lab/linearlearner_validation/{}'.format(validation_file)).upload_fileobj(f)
         >>> validate_recordIO_protobuf_location = 's3://{}/algorithms_lab/linearlearner_validation/{}'.format(bucket, validation_file)
         >>> print('The Pipe mode recordIO protobuf validation data: {}'.format(validate_recordIO_protobuf_location))

             The Pipe mode recordIO protobuf validation data: s3://modeling-ufo-lab1/algorithms_lab/linearlearner_validation/ufo_sightings_validatioin_recordIO_protobuf.data

         >>> # Alright we are good to go for the Linear Learner algorithm. Let's get everything we need from the ECR repository to call
         >>> #   the Linear Learner algorithm.



         >>> from sagemaker import image_uris
         >>> container = image_uris.retrieve('linear-learner', boto3.Session().region_name, '1')

         >>> container
             '382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:1'


         >>> # Create a training job name
         >>> job_name = 'ufo-linear-learner-job-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
         >>> print('Here is the job name {}'.format(job_name))
         >>>
         >>> # Here is where the model-artifact will be stored
         >>> output_location = 's3://{}/algorithms_lab/linearlearner_output'.format(bucket)

             Here is the job name ufo-linear-learner-job-20240701194728


         >>> # Next we start building out our model by using the SageMaker Python SDK and passing in everything that is
         >>> #   required to create a Linear Learner model.
         >>> #
         >>> # First I like to always create a specific job name.
         >>> #
         >>> # Next, we'll need to specify training parameters.
         >>>
         >>> # 1. The linear-learner algorithm container
         >>> # 2. The IAM role to use
         >>> # 3. Training instance type and count
         >>> # 4. S3 location for output data/model artifact
         >>> # 5. The input type (Pipe)
         >>> #     https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html
         >>> # 6. Linear Learner Hyperparameters
         >>> #     https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html
         >>> #
         >>> # Finally, after everything is included and ready, then we can call the .fit() function which specifies the S3 location
         >>> #  for training and validation data.

         >>> data_train.shape
             (14430, 23)

         >>> print('The feature_dim hyperparameter needs to be set to {}.'.format(data_train.shape[1] - 1))

             The feature_dim hyperparameter needs to be set to 22.


         >>> sess = sagemaker.Session()
         >>>
         >>> # Setup the LinearLeaner algorithm from the ECR container
         >>> linear = sagemaker.estimator.Estimator(container, role, instance_count=1, instance_type='ml.c4.xlarge',
         >>>                                        output_path=output_location, sagemaker_session=sess, input_mode='Pipe')
         >>> # Setup the hyperparameters
         >>> linear.set_hyperparameters(feature_dim=22, # number of attributes (minus the researchOutcome attribute)
         >>>                            predictor_type='multiclass_classifier', # type of classification problem
         >>>                            num_classes=3)  # number of classes in out researchOutcome (explained, unexplained, probable)
         >>>
         >>>
         >>> # Launch a training job. This method calls the CreateTrainingJob API call
         >>> data_channels = { 'train': training_recordIO_protobuf_location, 'validation': validate_recordIO_protobuf_location }
         >>> linear.fit(data_channels, job_name=job_name)


             INFO:sagemaker:Creating training-job with name: ufo-linear-learner-job-20240701194728
             2024-07-01 19:47:36 Starting - Starting the training job...
             2024-07-01 19:47:51 Starting - Preparing the instances for training...
             2024-07-01 19:48:19 Downloading - Downloading input data...
             2024-07-01 19:48:59 Downloading - Downloading the training image.........
             2024-07-01 19:50:20 Training - Training image download completed. Training in progress..Docker entrypoint called with argument(s): train
              . . .
              . . .
             [07/01/2024 19:50:55 INFO 139723470354240] #validation_score (algo-1) : ('multiclass_balanced_accuracy', 0.9270477490089298)
             [07/01/2024 19:50:55 INFO 139723470354240] #validation_score (algo-1) : ('multiclass_log_loss', 0.6215193271011654)
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation multiclass_cross_entropy_objective <loss>=0.177802592028485
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation multiclass_accuracy <score>=0.9469825155104343
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation multiclass_top_k_accuracy_3 <score>=1.0
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation dcg <score>=0.978291278630411
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation macro_recall <score>=0.9270477294921875
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation macro_precision <score>=0.9028711915016174
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation macro_f_1.000 <score>=0.9140763282775879
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation multiclass_balanced_accuracy <score>=0.9270477490089298
             [07/01/2024 19:50:55 INFO 139723470354240] #quality_metric: host=algo-1, validation multiclass_log_loss <score>=0.6215193271011654
             [07/01/2024 19:50:55 INFO 139723470354240] Best model found for hyperparameters: {"optimizer": "adam", "learning_rate": 0.03578924546103785, "l1": 0.3730851993396367, "wd": 0.0020367449461466797, "lr_scheduler_step": 153, "lr_scheduler_factor": 0.9895518165671255, "lr_scheduler_minimum_lr": 2.294182951069527e-05}
             [07/01/2024 19:50:55 INFO 139723470354240] Saved checkpoint to "/tmp/tmpxy_r1r8a/mx-mod-0000.params"
             [07/01/2024 19:50:55 INFO 139723470354240] Test data is not provided.
             #metrics {"StartTime": 1719863432.4299383, "EndTime": 1719863455.1427574, "Dimensions": {"Algorithm": "Linear Learner", "Host": "algo-1", "Operation": "training"}, "Metrics": {"initialize.time": {"sum": 1342.7329063415527, "count": 1, "min": 1342.7329063415527, "max": 1342.7329063415527}, "epochs": {"sum": 15.0, "count": 1, "min": 15, "max": 15}, "check_early_stopping.time": {"sum": 7.383584976196289, "count": 9, "min": 0.16188621520996094, "max": 4.042625427246094}, "update.time": {"sum": 18203.925371170044, "count": 8, "min": 2251.8746852874756, "max": 2326.157569885254}, "finalize.time": {"sum": 2108.978033065796, "count": 1, "min": 2108.978033065796, "max": 2108.978033065796}, "setuptime": {"sum": 1.8444061279296875, "count": 1, "min": 1.8444061279296875, "max": 1.8444061279296875}, "totaltime": {"sum": 22805.270433425903, "count": 1, "min": 22805.270433425903, "max": 22805.270433425903}}}

             2024-07-01 19:51:08 Completed - Training job completed
             Training seconds: 170
             Billable seconds: 170

         >>> print('Here is the location of the trained Linear Learner model: {}/{}/output/model.tar.gz'.format(output_location, job_name))

             Here is the location of the trained Linear Learner model: s3://modeling-ufo-lab1/algorithms_lab/linearlearner_output/ufo-linear-learner-job-20240701194728/output/model.tar.gz

             # Note: Linear Learner model had: multiclass_accuracy <score>=0.9469825155104343


             From here we have two trained models to present to Mr. K. Congratulations!

