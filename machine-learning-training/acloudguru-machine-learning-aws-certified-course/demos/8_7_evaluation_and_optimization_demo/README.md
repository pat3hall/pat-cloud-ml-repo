8.7 Demo: Evaluation and Optimization

  Resources:

    Hyperparameter Tuning with Amazon SageMaker's Automatic Model Tuning - AWS Online Tech Talks
      https://www.youtube.com/watch?v=ynYnZywayC4

    Linear Learning Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html


    Note: Downloaded demo files to:
      C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\8_7_evaluation_and_optimization_demo

    Training Dataset in recordIO protobuf data (ufo_sightings_validatioin_recordIO_protobuf.data)
      https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter8/ufo_sightings_train_recordIO_protobuf.data

    Validation Dataset in recordIO protobuf data (ufo_sightings_validatioin_recordIO_protobuf.data)
      https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter8/ufo_sightings_validatioin_recordIO_protobuf.data

    Jupyter Notebook (ufo-evaluation-optimization-lab.ipyynb
    https://github.com/ACloudGuru-Resources/Course_AWS_Certified_Machine_Learning/blob/master/Chapter8/ufo-evaluation-optimization-lab.ipynb


    ------------------------------------------------------

  Sending out Team of Scientist Overview

    - Mr K has g given us the go to use the Linear Learner model for identifying the legitimacy
      of a reported UFO sighting.
    - He plans to send out a team of scientists if any reported UFO sighting is probable or unexplained.

    - Before deploying the model, he wants us to make sure that we have the most optimized model,
      improve any performance and training and possibly improve the accuracy.

  Sending out Team of Scientist Steps / Questions to resolve:

    - tune our model to find the most optimized model for our problem
    - determine if the model is less accurate, more accurate, or about the same
    - What is the objective metric woudl you want to monitor to ensure this? How do you plan on
      measuring success?
    - Which hyperparameters need to be tuned? What combinations of hyperparameters need to be used?
      Note: previously, we used the default hyperparameter values
    - How much faster was training time improved [compared to baseline model from previous demo]?

  Final Results
    - best training job hyperparameters
    - difference in time between baseline model and optimized model
    - difference in accuracy between models

  Use SageMaker Hyperparameter Tuning job

    - create a SageMaker Hyperparameter tuning job with different ranges of values for hyperparameters
      to find the best configuration to minimize the validation:objective_loss metric


   validation:objective_loss
     https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner-tuning.html
     - The mean value of the objective loss function on the validation dataset every epoch.
     - By default, the loss is logistic loss for binary classification and squared loss for regression.
     - To set loss to other types, use the loss hyperparameter.

    objective_loss metric
      - this metric is used because this is what's used in multi classification problems.
      - this metric measures the performance of our classification model.
      - repeatedly calculate the difference between the values that our model is predicting
        and the actual values of the label.
      - AWS recommends that we minimize this value when using it as our objective metric.


    ------------------------------------------------------


  AWS Console -> SageMaker
     -> Training <left tab> -> Hyperparameter tuning job ->  Create hyperparameter tuning job ->
       Hyperparameter tuning job name: linear-learner-tuning-job -> Next ->
       -> Add training job definition (must be unique in your AWS account): linear-learning-tuning-070924,
          IAM Role: AmazonSageMaker-ExecutionRole-*,
          Algoritm Options: Built-in, Choose an Algorithm: Tabular - LinearbLearner,
          input mode: Pipe, Objective Metric: validation:objective_loss, type: minimize
          Hyperparameter Configuration:
              Name            Type           Scaling Type     Value / Range
              feature_dim     static                           22
              min_batch_size  Integer        Linear            500  - 5000
              epochs          static                           15
              predictor_type  static                           multiclass_classifier
              wd              Continuous     Logarithmic       .0001  - 1.0
              l1              Continuous     Logarithmic       .0001  - 1.0
              learning_rate   Continuous     Logarithmic       .0001  - 1.0
              num_classes     static                           3

           -> Next -> Channels -> Channel name: train, Input mode: Pipe, Data Source: S3,
              S3 Location: s3://modeling-ufo-lab1/algorithms_lab/linearlearner_train/ufo_sightings_train_recordIO_protobuf.data

              -> Add Channel:
                Channel name: validation, Input mode: Pipe, Data Source: S3,
              S3 Location: s3://modeling-ufo-lab1/algorithms_lab/linearlearner_validation/ufo_sightings_validatioin_recordIO_protobuf.data

            Output data Configuraiton:
              S3 output path: s3://modeling-ufo-lab1/optimization_evaluation_lab/hyperparameter_tuning_output

            -> Next ->

            Resource Configurations: Instance type: ml.c5.xlarge, Instance count: 1
               Stopping Condition: Maximum duration per training job: 20 min

            -> Next ->  <review Training Job definition -> Next ->

            Resource Limits:
              Maximum training jobs: 50,
              Maximum parallel training jobs: 5


           -> Create hyperparameter tuning job
              ->  took 16 min to complete


    Best training job summary:
    Name                                        Status          Objective metric                value
    linear-learning-tuning-job-030-eda920d9     Completed       validation:objective_loss       0.1782093346118927

    Best training job hyperparameters:

	Name            Type    Value
	l1	        -	0.0005953224641699573
	learning_rate	-	0.04380096472026997
	mini_batch_size	-	2977
	use_bias	-	true
	wd	        -	0.0019134739491125144

      # search for best training job in cloudWatch

        AWS Console -> CloudWatch -> log -> Log Grous -> /aws/sagemaker/TrainingJobs ->
          Search: linear-learning-tuning-job-030-eda920d9 -> select

        [07/09/2024 22:20:57 INFO 139932632377152] #quality_metric: host=algo-1, validation multiclass_accuracy <score>=0.9469825155104343
        . . .
	[07/09/2024 22:20:57 INFO 139932632377152] #quality_metric: host=algo-1, validation macro_recall <score>=0.9270477294921875
	[07/09/2024 22:20:57 INFO 139932632377152] #quality_metric: host=algo-1, validation macro_precision <score>=0.9028711915016174
	[07/09/2024 22:20:57 INFO 139932632377152] #quality_metric: host=algo-1, validation macro_f_1.000 <score>=0.9140763282775879

          examine best job log to find:  accuracy, f1, precision, recall:

           validation multiclass_accuracy <score>=0.9469825155104343
           validation macro_recall <score>=0.9270477294921875
           validation macro_precision <score>=0.9028711915016174
           validation macro_f_1.000 <score>=0.9140763282775879


          previous (7.14 Demo: Algorithms) . to find accuracy, f1, precision, recall:
            multiclass_accuracy <score>=0.9469825155104343
            macro_recall <score>=0.9270477294921875
            macro_precision <score>=0.9028711915016174
            macro_f_1.000 <score>=0.9140763282775879

            -> identical results! (in video, accuracy was slightly improved)

      # re- train using hyperparameters from best model

   AWS Console -> SageMaker -> Notebook -> Notebook Instances -> select "my-notebook-inst" -> start
   # when running:
   -> Open Jupyter
     -> Upload "ufo-evaluation-optimization-lab.ipynb"
        -> start uploaded notebook



code:
         >>> # First let's go ahead and import all the needed libraries.

         >>> import pandas as pd
         >>> import numpy as np
         >>> from datetime import datetime
         >>>
         >>> import boto3
         >>> from sagemaker import get_execution_role
         >>> import sagemaker

         >>> role = get_execution_role()
         >>> bucket='modeling-ufo-lab1'

         >>> # 1. Create and train our "optimized" model (Linear Learner)
         >>>
         >>> # Let's evaluate the Linear Learner algorithm with the new optimized hyperparameters.
         >>> # Let's go ahead and get the data that we already stored into S3 as recordIO protobuf data.


         >>> # Let's get the recordIO file for the training data that is in S3

         >>> train_file = 'ufo_sightings_train_recordIO_protobuf.data'
         >>> training_recordIO_protobuf_location = 's3://{}/algorithms_lab/linearlearner_train/{}'.format(bucket, train_file)
         >>> print('The Pipe mode recordIO protobuf training data: {}'.format(training_recordIO_protobuf_location))

             The Pipe mode recordIO protobuf training data: s3://modeling-ufo-lab1/algorithms_lab/linearlearner_train/ufo_sightings_train_recordIO_protobuf.data

         >>> # Let's get the recordIO file for the validation data that is in S3

         >>> validation_file = 'ufo_sightings_validatioin_recordIO_protobuf.data'
         >>> validate_recordIO_protobuf_location = 's3://{}/algorithms_lab/linearlearner_validation/{}'.format(bucket, validation_file)
         >>> print('The Pipe mode recordIO protobuf validation data: {}'.format(validate_recordIO_protobuf_location))

             The Pipe mode recordIO protobuf validation data: s3://modeling-ufo-lab1/algorithms_lab/linearlearner_validation/ufo_sightings_validatioin_recordIO_protobuf.data


         >>> # Alright we are good to go for the Linear Learner algorithm. Let's get everything we need from the ECR repository to call
         >>> #    the Linear Learner algorithm.

         >>> from sagemaker import image_uris
         >>> container = image_uris.retrieve('linear-learner', boto3.Session().region_name, '1')


         >>> # Let's create a job and use the optimzed hyperparameters.

         >>> # Create a training job name
         >>> job_name = 'ufo-linear-learner-job-optimized-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
         >>> print('Here is the job name {}'.format(job_name))
         >>>
         >>> # Here is where the model-artifact will be stored
         >>> output_location = 's3://{}/optimization_evaluation_lab/linearlearner_optimized_output'.format(bucket)

             Here is the job name ufo-linear-learner-job-optimized-20240709233014

         >>> # Next we can start building out our model by using the SageMaker Python SDK and passing in everything that is
         >>> #   required to create a Linear Learner training job.
         >>>
         >>> # Here are the linear learner hyperparameters that we can use within our training job.
         >>> #     https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html

         >>> # After we run this job we can view the results.

         >>> # note: video shows adding more parameters, I only the tuned hyperparameters

         >>> %%time
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
         >>> linear.set_hyperparameters( feature_dim=22,
         >>>                             predictor_type='multiclass_classifier',
         >>>                             num_classes=3,
         >>>                             l1=0.0005953224641699573,
         >>>                             learning_rate=0.04380096472026997,
         >>>                             mini_batch_size=2977,
         >>>                             use_bias='true',
         >>>                             wd=0.0019134739491125144
         >>>                           )
         >>>
         >>>
         >>> # Launch a training job. This method calls the CreateTrainingJob API call
         >>> data_channels = {
         >>>     'train': training_recordIO_protobuf_location,
         >>>     'validation': validate_recordIO_protobuf_location
         >>> }
         >>> linear.fit(data_channels, job_name=job_name)


             INFO:sagemaker:Creating training-job with name: ufo-linear-learner-job-optimized-20240709233014

             2024-07-09 23:42:01 Starting - Starting the training job...
             2024-07-09 23:42:17 Starting - Preparing the instances for training...
             2024-07-09 23:42:47 Downloading - Downloading input data...
             2024-07-09 23:43:28 Downloading - Downloading the training image.........
             2024-07-09 23:44:44 Training - Training image download completed. Training in progress.Docker entrypoint called with argument(s): train
             . . .

             [07/09/2024 23:45:36 INFO 140579665631040] #quality_metric: host=algo-1, validation multiclass_accuracy <score>=0.9469825155104343
             [07/09/2024 23:45:36 INFO 140579665631040] #quality_metric: host=algo-1, validation multiclass_top_k_accuracy_3 <score>=1.0
             [07/09/2024 23:45:36 INFO 140579665631040] #quality_metric: host=algo-1, validation dcg <score>=0.978291244205619
             [07/09/2024 23:45:36 INFO 140579665631040] #quality_metric: host=algo-1, validation macro_recall <score>=0.9270477294921875
             [07/09/2024 23:45:36 INFO 140579665631040] #quality_metric: host=algo-1, validation macro_precision <score>=0.9028711915016174
             [07/09/2024 23:45:36 INFO 140579665631040] #quality_metric: host=algo-1, validation macro_f_1.000 <score>=0.9140763282775879
             [07/09/2024 23:45:36 INFO 140579665631040] #quality_metric: host=algo-1, validation multiclass_balanced_accuracy <score>=0.9270477490089298
             [07/09/2024 23:45:36 INFO 140579665631040] #quality_metric: host=algo-1, validation multiclass_log_loss <score>=0.6202281367401643
             [07/09/2024 23:45:36 INFO 140579665631040] Best model found for hyperparameters: {"optimizer": "adam", "learning_rate": 0.04380096472026997, "l1": 0.0005953224641699573, "wd": 0.0019134739491125144, "lr_scheduler_step": 10, "lr_scheduler_factor": 0.98, "lr_scheduler_minimum_lr": 1e-05}
             [07/09/2024 23:45:36 INFO 140579665631040] Saved checkpoint to "/tmp/tmp67b8ckvo/mx-mod-0000.params"
             [07/09/2024 23:45:36 INFO 140579665631040] Test data is not provided.
             #metrics {"StartTime": 1720568698.8876836, "EndTime": 1720568736.392592, "Dimensions": {"Algorithm": "Linear Learner", "Host": "algo-1", "Operation": "training"}, "Metrics": {"initialize.time": {"sum": 1422.2440719604492, "count": 1, "min": 1422.2440719604492, "max": 1422.2440719604492}, "epochs": {"sum": 15.0, "count": 1, "min": 15, "max": 15}, "check_early_stopping.time": {"sum": 12.825965881347656, "count": 16, "min": 0.29158592224121094, "max": 1.1529922485351562}, "update.time": {"sum": 32834.91826057434, "count": 15, "min": 2167.7138805389404, "max": 2247.562885284424}, "finalize.time": {"sum": 2165.846347808838, "count": 1, "min": 2165.846347808838, "max": 2165.846347808838}, "setuptime": {"sum": 1.8286705017089844, "count": 1, "min": 1.8286705017089844, "max": 1.8286705017089844}, "totaltime": {"sum": 37600.07357597351, "count": 1, "min": 37600.07357597351, "max": 37600.07357597351}}}


             2024-07-09 23:45:53 Uploading - Uploading generated training model
             2024-07-09 23:45:53 Completed - Training job completed
             Training seconds: 185
             Billable seconds: 185
             CPU times: user 905 ms, sys: 68.5 ms, total: 974 ms
             Wall time: 4min 13s


         >>> # Now we can compare the amount of time billed and the accuracy compared to our baseline model.

            -> previous billable secords was 119 (new job is slower!)
    ------------------------------------------------------


    -> stop notebook instance

