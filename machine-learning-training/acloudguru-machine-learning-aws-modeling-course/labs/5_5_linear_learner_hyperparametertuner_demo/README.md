------------------------------------------------------
5.5 Optimizing Hyperparameters in a Linear Model


  Saved files:
    completed python jupyter notebook:
      Linear_Learner_Regression_csv_format.ipynb
    extracted python code from jupyter notebook:
      Linear_Learner_Regression_csv_format.py
    html view from completed jupyter notebook:
      Linear_Learner_Regression_csv_format.html


   Choosing the right combination of hyperparameters
     - is critical for the efficiency and effectiveness of the optimization process.
     - finding the right combinatin of hyperparameter values will help the model predict accurately

     Hyperparameter optimization
        - previously learned the different strategies used to tune the hyperparameters like grid search and random search.
        Gradient Descent algorithm
          - also can use gradient descent to find optimal hyperparameter values.
          - In this approach, the hyperparameters are adjusted by computing the gradients of the objective function
            with respect to these hyperparameters and updating them iteratively.

     Effect or the learning rate
       small learning rate
         - With a slow learning rate, the algorithm makes tiny updates to the parameters needing a large number
           of iterations to converge.
         - The algorithm may also get stuck in a local minimum, especially in a non convex problem.
       large learning rate
         - A large learning rate may overshoot the minimum, causing the parameters to oscillate around the minimum
           and never converging making the training process unstable.

     How to define hyperparameters
       - during the tuning process using a parameter range.
       parameter range
         categorical parameter range.
           - You define the different categories of hyperparameter values that you want the tuning job to select.
            "CategoricalParameterRanges":
             [
               {
                  "Name": "tree-method",
                  "Values": ["auto", "exact", "approx"]
               }
             ]

         continuous parameter range.
           - You specify the minimum and the maximum value of the hyperparameter range, and the tuning job will
             select a value between this range during the tuning process.

            "ContinuousParameterRages":
             [
               {
                  "Name": "eta",
                  "MaxValues": 0.5, "MinValue": 0, "ScalingType": "Auto"
               }
             ]


         integer parameter range.
           - You specify the minimum and the maximum hyperparameter range value as well, but the values need to
             be an integer value.

            "IntegerParameterRages":
             [
               {
                  "Name": "max_depth",
                  "MaxValues": 10, "MinValue": 1, "ScalingType": "Auto"
               }
             ]


      Hyperparameter scaling Types
        auto.
          - SageMaker hyperparameter tuning chooses the best scale for the hyperparameter.
        linear.
          - The tuning job selects the value in a linear fashion starting from lowest to the highest incrementing in smaller intervals.
        logarithmic.
          - The scaling works only for positive values.
          - Use logarithmic scaling when you are searching a range that spans a several order of magnitude.
        reverse logarithmic.
          - This is supported only in continuous parameter range only and not in integer parameter range.
          - It works only for ranges that are between zero and one,
          - choose this option when you are working on a range that is highly sensitive to small changes.

   Note: Missing - Link to demo notebook - appear to be the below:
   Demo: Optimiziation hyperparameters of a linear learning using AMT

        Using the MNIST dataset, we train a binary classifier to predict a single digit - includes AMT option.
          https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist.html
             saved to: notebooks/Linear_Learner_Regression_csv_format.ipynb

         also at:

           SageMaker Notebook instance -> Jupyter -> SageMaker Examples -> Introduction to Amazon Algorithms
               -> Linear_Learner_Regression_csv_format.ipynb-> Use


    Linear Learning Hyperparameters in demo:
      wd
        - The weight decay parameter, also known as the L2 regularization parameter.
        - If you don't want to use L2 regularization, set the value to 0.
        - Optional
        - Valid values:auto or non-negative floating-point integer
        - Default value: auto

      learning_rate
        - The step size used by the optimizer for parameter updates.
        - Optional
        - Valid values: auto or positive floating-point integer
        - Default value: auto, whose value depends on the optimizer chosen.

      mini_batch_size
       - The number of observations per mini-batch for the data iterator.
       - Optional
       - Valid values: Positive integer
       - Default value: 1000

    Code: Linear Linear Demo Automatic Model Tuning code only

          Training with Automatic Model Tuning (HPO)

          As mentioned above, instead of manually configuring our hyper parameter values and training with SageMaker Training,
          we'll use Amazon SageMaker Automatic Model Tuning.

          The code sample below shows you how to use the HyperParameterTuner. For recommended default hyparameter ranges,
          check the Amazon SageMaker Linear Learner HPs documentation.

          The tuning job will take 8 to 10 minutes to complete.

      >>> import time
      >>> from sagemaker.tuner import IntegerParameter, ContinuousParameter
      >>> from sagemaker.tuner import HyperparameterTuner

      >>> job_name = "DEMO-ll-aba-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
      >>> print("Tuning job name: ", job_name)

      >>> # Linear Learner tunable hyper parameters can be found here https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner-tuning.html
      >>> hyperparameter_ranges = {
      >>>     "wd": ContinuousParameter(1e-7, 1, scaling_type="Auto"),
      >>>     "learning_rate": ContinuousParameter(1e-5, 1, scaling_type="Auto"),
      >>>     "mini_batch_size": IntegerParameter(100, 2000, scaling_type="Auto"),
      >>> }

      >>> # Increase the total number of training jobs run by AMT, for increased accuracy (and training time).
      >>> max_jobs = 6
      >>> # Change parallel training jobs run by AMT to reduce total training time, constrained by your account limits.
      >>> # if max_jobs=max_parallel_jobs then Bayesian search turns to Random.
      >>> max_parallel_jobs = 2

      >>> hp_tuner = HyperparameterTuner(
      >>>     linear,
      >>>     "validation:mse",
      >>>     hyperparameter_ranges,
      >>>     max_jobs=max_jobs,
      >>>     max_parallel_jobs=max_parallel_jobs,
      >>>     objective_type="Minimize",
      >>> )

      >>> # Launch a SageMaker Tuning job to search for the best hyperparameters
      >>> hp_tuner.fit(inputs={"train": train_data, "validation": validation_data}, job_name=job_name)



