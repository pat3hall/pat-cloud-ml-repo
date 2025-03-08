9.9 Demo: Implementation and Operations Lab

  Resources:
    Call an Amazon SageMaker model endpoint using Amazon API Gateway and AWS Lambda
      https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/

    Note: Downloaded demo files to:
      C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\9_10_implementation_and_operations_demo

    Jupyter Notebook: ufo-implementation-operations-lab.ipynb
      https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter9/ufo-implementation-operations-lab.ipynb

    Lambda Function: lambda_function.py
      https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter9/lambda_function.py

    Sample Request JSON: sample_request.json
      https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter9/sample_request.json

    UFO Full Dataset: ufo_fullset.csv
      https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter9/ufo_fullset.csv


  Deploy Model into Production
    - Mr K has give us the go to deploy the optimized Linear Learner model in production
    - Once this ic complete, Mr K's team can use it to investigate any newly reported UFO sightings

    - our goal is deploy the model into production and give Mr K's team some way to interact with the deployed model

  Deploy Model into Production
    - deploy our Linear Learner model using SageMaker hosting
    - create a way to interact with the SageMaker endpoint created for the deployed model
    - How will Mr K's team interact with the model? Will there be some type of user interface?
    - What does the input and output of a request look like? Will it be in batches of immediate response?

  Final Results
    - Make a request to the SageMaker hosted model
        {
         "data": "45.0, 10.0,38.5816667,-121.49333329999999,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0"
        }
          Note: input data had to be transformed to numeric format for the Linear learner model
    - view the output response from the trained model:
          Explained, Unexplained, or Probable

  Deploy to SageMaker Hosting
    - Simply call the deploy() method after training our model to deploy it to SageMaker hosting

    Notes:
        - reason I suggest using the .deploy method() is because when we use it within a Jupyter notebook, it makes
          it really easy and only one line of code to deploy our model.
        - if you're using the user interface, you could simply point it to the model artifact that you already trained,
          or you could point it to a training job that you had pre created.

  Create Lambda and API GateWay Endpoint
    - Create an API Gateway endpoint that invokes a Lambda function
    - This Lambda function calls '.invoke_endpoint()' method with an new UFO sightings (calls the sagemaker trained model)
    - return the results from the models prediction

   SageMaker Hosting, Lambda, API Gateway
      - create a new Jupyter notebook and retrain our model.
      - when retraining our model, so we can hold out a portion of our data set to test it once we deploy our model.
      - after retraining, going to then deploy it to a SageMaker model endpoint and then use our test data set to test
        against that newly created endpoint
      -  Next, create a Lambda function which calls the .invoke_endpoint() method and sends a request of a new UFO sighting.
      - Once we have our Lambda function set up, we can then create an API Gateway endpoint, which creates an HTTP endpoint
        that we can call from API Gateway, or we can call it from any other application.

                                                                        |----------------------------------|
          UFO         HTTPS Endpoint         Lambda Function            |SageMaker Model Endpoint          |
        Sighting ---> API Gateway     ---->  .invoke_endpoint() ------> |                                  |
          input       Endpoint                                          |             Notebook             |
                                                                        |             .deploy()            |
                                                                        | Train                       Model|
                                                                        |----------------------------------|

   ------------------------------------------------------
   AWS Console -> SageMaker -> Notebook -> Notebook Instances -> select "my-notebook-inst" -> start
   # when running:
   -> Open Jupyter
     -> Upload "ufo-implementation-operations-lab.ipynb"
        -> start uploaded notebook



    code:  UFO Implementation and Operations Lab

         >>> # First let's go ahead and import all the needed libraries.
         >>> import pandas as pd
         >>> import numpy as np
         >>> from datetime import datetime
         >>>
         >>> # import module in terms of dealing with various types of I/O
         >>> import io
         >>>
         >>> # import sagemaker common library
         >>> import sagemaker.amazon.common as smac
         >>>
         >>> import boto3
         >>> from sagemaker import get_execution_role
         >>> import sagemaker
         >>>
         >>> import matplotlib.pyplot as plt
         >>> import seaborn as sns

         >>> # Step 1: Loading the data from Amazon S3¶
         >>> # Let's get the UFO sightings data that is stored in S3 and load it into memory.

         >>> role = get_execution_role()
         >>> bucket='modeling-ufo-lab1'
         >>> sub_folder = 'ufo_dataset'
         >>> data_key = 'ufo_fullset.csv'
         >>> data_location = 's3://{}/{}/{}'.format(bucket, sub_folder, data_key)

             df = pd.read_csv(data_location, low_memory=False)
             df.head()

              	reportedTimestamp 	eventDate 	eventTime 	shape 	duration 	witnesses 	weather 	firstName 	lastName 	latitude 	longitude 	sighting 	physicalEvidence 	contact 	researchOutcome
             0 	1977-04-04T04:02:23.340Z 	1977-03-31 	23:46 	circle 	4 	1 	rain 	Ila 	Bashirian 	47.329444 	-122.578889 	Y 	N 	N 	explained
             1 	1982-11-22T02:06:32.019Z 	1982-11-15 	22:04 	disk 	4 	1 	partly cloudy 	Eriberto 	Runolfsson 	52.664913 	-1.034894 	Y 	Y 	N 	explained
             .  . .

             Step 2: Cleaning, transforming and preparing the dataset

             This step is so important. It's crucial that we clean and prepare our data before we do anything else.

             Let's go ahead and start preparing our dataset by transforming some of the values into the correct data types.
             Here is what we are going to take care of.

              1. Convert the reportedTimestamp and eventDate to a datetime data types.
              2. Convert the shape and weather to a category data type.
              3. Map the physicalEvidence and contact from 'Y', 'N' to 0, 1.
              4. Convert the researchOutcome to a category data type (target attribute).

             Let's also drop the columns that are not important.

              1. We can drop sighting becuase it is always 'Y' or Yes.
              2. Let's drop the firstName and lastName becuase they are not important in determining the researchOutcome.
              3. Let's drop the reportedTimestamp becuase when the sighting was reporting isn't going to help us determine
                 the legitimacy of the sighting.
              4. We would need to create some sort of buckets for the eventDate and eventTime, like seasons for example, but
                 since the distribution of dates is pretty even, let's go ahead and drop them.

             Finally, let's apply one-hot encoding

              1. We need to one-hot both the weather attribute and the shape attribute.
              2. We also need to transform or map the researchOutcome (target) attribute into numeric values. This is what the
                alogrithm is expecting. We can do this by mapping unexplained, explained, and probable to 0, 1, 2.


         >>> # Replace the missing values with the most common shape (circle)
         >>> df['shape'] = df['shape'].fillna(df['shape'].value_counts().index[0])
         >>>
         >>> # Convert the reportedTimestamp and eventDate to a datetime data types.
         >>> df['reportedTimestamp'] = pd.to_datetime(df['reportedTimestamp'])
         >>> df['eventDate'] = pd.to_datetime(df['eventDate'])
         >>>
         >>> # Convert the shape and weather to a category data type.
         >>> df['shape'] = df['shape'].astype('category')
         >>> df['weather'] = df['weather'].astype('category')
         >>>
         >>> # Map the physicalEvidence and contact from 'Y', 'N' to 0, 1
         >>> df['physicalEvidence'] = df['physicalEvidence'].replace({'Y': 1, 'N': 0})
         >>> df['contact'] = df['contact'].replace({'Y': 1, 'N': 0})
         >>>
         >>> # Convert the researchOutcome to a category data type (target attribute).
         >>> df['researchOutcome'] = df['researchOutcome'].astype('category')
         >>>
         >>> # We can drop sighting becuase it is always 'Y' or Yes.
         >>> # Let's drop the firstName and lastName becuase they are not important in determining
         >>> #    the researchOutcome.
         >>> # Let's drop the reportedTimestamp becuase when the sighting was reporting isn't going
         >>> #   to help us determine the legitimacy of the sighting.
         >>> df.drop(columns=['firstName', 'lastName', 'sighting', 'reportedTimestamp', 'eventDate', 'eventTime'], inplace=True)
         >>>
         >>> # Let's one-hot the weather and shape attribute
         >>> # Note: FIXED! needed to add 'dtype='int' because get_dummies dtype default is now 'bool'
         >>> df = pd.get_dummies(df, columns=['weather', 'shape'], dtype='int')
         >>>
         >>> # Let's replace the researchOutcome values with 0, 1, 2 for Unexplained, Explained, and Probable
         >>> df['researchOutcome'] = df['researchOutcome'].replace({'unexplained': 0, 'explained': 1, 'probable': 2})

         >>> display(df.head())
         >>> display(df.shape)

              	duration 	witnesses 	latitude 	longitude 	physicalEvidence 	contact 	researchOutcome 	weather_clear 	weather_fog 	weather_mostly cloudy 	... 	weather_stormy 	shape_box 	shape_circle 	shape_disk 	shape_light 	shape_oval 	shape_pyramid 	shape_sphere 	shape_square 	shape_triangle
             0 	4 	1 	47.329444 	-122.578889 	0 	0 	1 	0 	0 	0 	... 	0 	0 	1 	0 	0 	0 	0 	0 	0 	0
             1 	4 	1 	52.664913 	-1.034894 	1 	0 	1 	0 	0 	0 	... 	0 	0 	0 	1 	0 	0 	0 	0 	0 	0
             . . .

             5 rows × 23 columns

             (18000, 23)


             Step 3: Creating and training our model (Linear Learner)

             Let's evaluate the Linear Learner algorithm as well. Let's go ahead and randomize the data again and get it ready
             for the Linear Leaner algorithm. We will also rearrange the columns so it is ready for the algorithm (it expects
             the first column to be the target attribute)


         >>> # random and split data (80% train, 10% valication, 10% test)
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
         >>> # Breaks the datasets into attribute numpy.ndarray and the same for target attribute.
         >>> train_X = data_train.drop(columns='researchOutcome').values
         >>> train_y = data_train['researchOutcome'].values
         >>>
         >>> val_X = data_val.drop(columns='researchOutcome').values
         >>> val_y = data_val['researchOutcome'].values
         >>>
         >>> test_X = data_test.drop(columns='researchOutcome').values
         >>> test_y = data_test['researchOutcome'].values

             Next, Let's create recordIO file for the training data and upload it to S3.


         >>> train_file = 'ufo_sightings_train_recordIO_protobuf.data'
         >>>
         >>> # converts the data in numpy array format to RecordIO format (using SageMaker common library)
         >>> f = io.BytesIO()
         >>> smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
         >>> f.seek(0)
         >>>
         >>> boto3.Session().resource('s3').Bucket(bucket).Object('implementation_operations_lab/linearlearner_train/{}'.format(train_file)).upload_fileobj(f)
         >>> training_recordIO_protobuf_location = 's3://{}/implementation_operations_lab/linearlearner_train/{}'.format(bucket, train_file)
         >>> print('The Pipe mode recordIO protobuf training data: {}'.format(training_recordIO_protobuf_location))
             The Pipe mode recordIO protobuf training data: s3://modeling-ufo-lab1/implementation_operations_lab/linearlearner_train/ufo_sightings_train_recordIO_protobuf.data

             Let's create recordIO file for the validation data and upload it to S3

         >>> validation_file = 'ufo_sightings_validatioin_recordIO_protobuf.data'
         >>>
         >>> f = io.BytesIO()
         >>> smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
         >>> f.seek(0)
         >>>
         >>> boto3.Session().resource('s3').Bucket(bucket).Object('implementation_operations_lab/linearlearner_validation/{}'.format(validation_file)).upload_fileobj(f)
         >>> validate_recordIO_protobuf_location = 's3://{}/implementation_operations_lab/linearlearner_validation/{}'.format(bucket, validation_file)
         >>> print('The Pipe mode recordIO protobuf validation data: {}'.format(validate_recordIO_protobuf_location))
             The Pipe mode recordIO protobuf validation data: s3://modeling-ufo-lab1/implementation_operations_lab/linearlearner_validation/ufo_sightings_validatioin_recordIO_protobuf.data


             Alright we are good to go for the Linear Learner algorithm. Let's get everything we need from the ECR repository to call the Linear Learner algorithm.


         >>> from sagemaker import image_uris
         >>> container = image_uris.retrieve('linear-learner', boto3.Session().region_name, '1')

         >>> # Create a training job name
         >>> job_name = 'ufo-linear-learner-job-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
         >>> print('Here is the job name {}'.format(job_name))
         >>>
         >>> # Here is where the model-artifact will be stored
         >>> output_location = 's3://{}/implementation_operations_lab/linearlearner_output'.format(bucket)
             Here is the job name ufo-linear-learner-job-20240717192831


             Next we start building out our model by using the SageMaker Python SDK and passing in everything that is required to create
               a Linear Learner model.

             First I like to always create a specific job name. Next, we'll need to specify training parameters.

             Finally, after everything is included and ready, then we can call the .fit() function which specifies the S3 location for
               training and validation data.

         >>> print('The feature_dim hyperparameter needs to be set to {}.'.format(data_train.shape[1] - 1))
             The feature_dim hyperparameter needs to be set to 22.

         >>> sess = sagemaker.Session()
         >>>
         >>> # Setup the LinearLeaner algorithm from the ECR container
         >>> linear = sagemaker.estimator.Estimator(container,
         >>>                                        role,
         >>>                                        instance_count=1,
         >>>                                        instance_type='ml.c4.xlarge',
         >>>                                        output_path=output_location,
         >>>                                        sagemaker_session=sess,
         >>>                                        input_mode='Pipe')
         >>> # Setup the hyperparameters
         >>> linear.set_hyperparameters(feature_dim=22,
         >>>                            predictor_type='multiclass_classifier',
         >>>                            num_classes=3,
         >>>                            # based on video hyperparameter tunning results
         >>>                            early_stopping_patience=3,
         >>>                            early_stopping_tolerance=0.001,
         >>>                            epochs=15, l1=0.0647741539306635,
         >>>                            learning_rate=0.09329042024421902,
         >>>                            loss='auto', mini_batch_size=744,
         >>>                            num_models='auto',
         >>>                            optimizer='auto',
         >>>                            unbias_data='auto',
         >>>                            unbias_label='auto',
         >>>                            use_bias='true',
         >>>                            wd=0.000212481391205101
         >>>                           )
         >>>
         >>> # Launch a training job. This method calls the CreateTrainingJob API call
         >>> data_channels = {
         >>>     'train': training_recordIO_protobuf_location,
         >>>     'validation': validate_recordIO_protobuf_location
         >>> }
         >>> linear.fit(data_channels, job_name=job_name)

             INFO:sagemaker:Creating training-job with name: ufo-linear-learner-job-20240717193437

             2024-07-17 19:34:50 Starting - Starting the training job...
             2024-07-17 19:35:05 Starting - Preparing the instances for training...
             2024-07-17 19:35:30 Downloading - Downloading input data...
             2024-07-17 19:36:04 Downloading - Downloading the training image.........
             2024-07-17 19:37:41 Training - Training image download completed. Training in progress..Docker entrypoint called with argument(s): train
             . . .
             [07/17/2024 19:38:23 INFO 139933004003136] #quality_metric: host=algo-1, validation multiclass_accuracy <score>=0.9396503102086858
             . . .
             [07/17/2024 19:38:23 INFO 139933004003136] Test data is not provided.
             #metrics {"StartTime": 1721245071.1513948, "EndTime": 1721245103.5236073, "Dimensions": {"Algorithm": "Linear Learner", "Host": "algo-1", "Operation": "training"}, "Metrics": {"initialize.time": {"sum": 1321.202039718628, "count": 1, "min": 1321.202039718628, "max": 1321.202039718628}, "epochs": {"sum": 15.0, "count": 1, "min": 15, "max": 15}, "check_early_stopping.time": {"sum": 6.572246551513672, "count": 13, "min": 0.1647472381591797, "max": 0.8738040924072266}, "update.time": {"sum": 27831.69913291931, "count": 12, "min": 2293.8809394836426, "max": 2408.231735229492}, "finalize.time": {"sum": 2153.674840927124, "count": 1, "min": 2153.674840927124, "max": 2153.674840927124}, "setuptime": {"sum": 1.8579959869384766, "count": 1, "min": 1.8579959869384766, "max": 1.8579959869384766}, "totaltime": {"sum": 32466.008186340332, "count": 1, "min": 32466.008186340332, "max": 32466.008186340332}}}


             2024-07-17 19:38:39 Uploading - Uploading generated training model
             2024-07-17 19:38:39 Completed - Training job completed
             Training seconds: 190
             Billable seconds: 190


         >>> print('Here is the location of the trained Linear Learner model: {}/{}/output/model.tar.gz'.format(output_location, job_name))
             Here is the location of the trained Linear Learner model: s3://modeling-ufo-lab1/implementation_operations_lab/linearlearner_output/ufo-linear-learner-job-20240717193437/output/model.tar.gz


             Step 4: Deploying the model into SageMaker hosting

             Next, let's deploy the model into SageMaker hosting onto a single m4 instance. We can then use this instance to test the model with the
               test data that we help out at the beginning of the notebook. We can then evaluate things like accuracy, precision, recall, and f1 score.

             We can use some fancy libraries to build out a confusion matrix/heatmap to see how accurate our model is.


         NOTE: linear.deploy() method does
                - spins up the specified type and number the instances (e.g. 1 ml.m4.xlarge instance) that our model is going to be deployed to,
                - creates the endpoint configuration
                - creates an endpoint for us on SageMaker hosting,


         >>> multiclass_predictor = linear.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
             INFO:sagemaker:Creating model with name: linear-learner-2024-07-17-20-51-48-106
             INFO:sagemaker:Creating endpoint-config with name linear-learner-2024-07-17-20-51-48-106
             INFO:sagemaker:Creating endpoint with name linear-learner-2024-07-17-20-51-48-106


            This next code is just setup code to allow us to draw out nice and pretty confusion matrix/heatmap.


         >>> from sklearn.metrics import confusion_matrix
         >>> from sklearn.utils.multiclass import unique_labels
         >>>
         >>> def plot_confusion_matrix(y_true, y_pred, classes,
         >>>                           normalize=False,
         >>>                           title=None,
         >>>                           cmap=None):
         >>>     """
         >>>     This function prints and plots the confusion matrix.
         >>>     Normalization can be applied by setting `normalize=True`.
         >>>     """
         >>>     if not title:
         >>>         if normalize:
         >>>             title = 'Normalized confusion matrix'
         >>>             plt.cm.Greens
         >>>         else:
         >>>             title = 'Confusion matrix, without normalization'
         >>>
         >>>     # Compute confusion matrix
         >>>     cm = confusion_matrix(y_true, y_pred)
         >>>     # Only use the labels that appear in the data
         >>>     classes = classes[unique_labels(y_true, y_pred)]
         >>>     if normalize:
         >>>         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
         >>> #         print("Normalized confusion matrix")
         >>> #     else:
         >>> #         print('Confusion matrix, without normalization')
         >>>
         >>> #     print(cm)
         >>>
         >>>     fig, ax = plt.subplots()
         >>>     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
         >>>     ax.figure.colorbar(im, ax=ax)
         >>>     # We want to show all ticks...
         >>>     ax.set(xticks=np.arange(cm.shape[1]),
         >>>            yticks=np.arange(cm.shape[0]),
         >>>            # ... and label them with the respective list entries
         >>>            xticklabels=classes, yticklabels=classes,
         >>>            title=title,
         >>>            ylabel='Actual',
         >>>            xlabel='Predicted')
         >>>
         >>>     # Rotate the tick labels and set their alignment.
         >>>     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         >>>              rotation_mode="anchor")
         >>>
         >>>     # Loop over data dimensions and create text annotations.
         >>>     fmt = '.2f' if normalize else 'd'
         >>>     thresh = cm.max() / 2.
         >>>     for i in range(cm.shape[0]):
         >>>         for j in range(cm.shape[1]):
         >>>             ax.text(j, i, format(cm[i, j], fmt),
         >>>                     ha="center", va="center",
         >>>                     color="white" if cm[i, j] > thresh else "black")
         >>>     fig.tight_layout()
         >>>     return ax
         >>>
         >>>
         >>> np.set_printoptions(precision=2)


         >>> # from sagemaker.predictor import json_deserializer, csv_serializer
         >>>
         >>> # multiclass_predictor.content_type = 'text/csv'
         >>> multiclass_predictor.serializer = sagemaker.serializers.CSVSerializer()
         >>> multiclass_predictor.deserializer = sagemaker.deserializers.JSONDeserializer()
         >>>
         >>> predictions = []
         >>> results = multiclass_predictor.predict(test_X)
         >>> predictions += [r['predicted_label'] for r in results['predictions']]
         >>> predictions = np.array(predictions)


         >>> %matplotlib inline
         >>> sns.set_context("paper", font_scale=1.4)
         >>>
         >>> y_test = test_y
         >>> y_pred = predictions
         >>>
         >>> class_names = np.array(['Unexplained', 'Explained', 'Probable'])
         >>>
         >>> # Plot non-normalized confusion matrix
         >>> plot_confusion_matrix(y_test, y_pred, classes=class_names,
         >>>                       title='Confusion matrix',
         >>>                       cmap=plt.cm.Blues)
         >>> plt.grid(False)
         >>> plt.show()


         >>> from sklearn.metrics import precision_recall_fscore_support
         >>> from sklearn.metrics import accuracy_score
         >>>
         >>> y_test = data_test['researchOutcome']
         >>> y_pred = predictions
         >>> scores = precision_recall_fscore_support(y_test, y_pred, average='macro', labels=np.unique(y_pred))
         >>> acc = accuracy_score(y_test, y_pred)
         >>> print('Accuracy is: {}'.format(acc))
         >>> print('Precision is: {}'.format(scores[0]))
         >>> print('Recall is: {}'.format(scores[1]))
         >>> print('F1 score is: {}'.format(scores[2]))
             Accuracy is: 0.9499165275459098
             Precision is: 0.9127438555359534
             Recall is: 0.9365168251900394
             F1 score is: 0.9241429254888969

   ------------------------------------------------------
   lambda function

   AWS Console -> Lambda Function -> Create Function -> select "author from scratch",
      Function Name: invoke-sagemaker-endpoint, Runtime: python: 3.8;
      select: "Create a new role with basic Lambda permissions"
      -> Create function

   role created: invoke-sagemaker-endpoint-role-7t49csfu

   AWS Console -> IAM -> Role -> select "invoke-sagemaker-endpoint-role-7t49csfu",
      -> Add permissions -> Attach Policy -> AmasonSageMakerFullAccess -> Add permissions


   insert "lambda_function.py" code

   -> Configuration <tab> -> Environment Variables <left tab> -> Edit -> Add environment variable ->
     Key: ENDPOINT_NAME  value: linear-learner-2024-07-17-20-51-48-106
     -> Save
     -> Deploy
     -> Test <center middle> -> Request Body:
         {
          "data": "45.0, 10.0,38.5816667,-121.49333329999999,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0"
        }
        -> Test

          / - POST method test results
          Request /
          Status 200
          Response body "Unexplained"

     -> Test <center middle> -> Request Body:
         {
          "data": "45.0,1.0,36.5816667,-121.49333329999999,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0
        }
        -> Test

          / - POST method test results
          Request /
          Status 200
          Response body "Explained"

          -> Deploy -> Deployment Staget: [New Stage], Stage name: production  -> Deploy

   --------------l----------------------------------------
   AWS console -> API Gateway -> Create API Gateway -> REST API -> Build
      "New API", API Name: ufo-inference-api, API endpoint type: Regional -> Create API

       Resources <left tab> -> Methods -> Create method -> Method Type: POST, Integration Type: Lambda,
         Lambda Function: *invoke-sagemaker-endpoint ->  create method

         Returns a Invoke URL


   ------------------------------------------------------



