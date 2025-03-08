------------------------------------------------------
3.10 Examining SageMaker's K Means Algorithm


  SageMaker K-means algorithm Overview
    - It's an unsupervised learning algorithm that groups similar data.
    - The similarity is determined based on the attributes we specify.
    - 'k' number must be predetermined for the algorithm.


   Amazon SageMaker
     - it uses a modified version of K-means algorithm and it is more accurate.
     - Expects tabular data
        - each row represents the observation, and
        - each column represents the attribute or the features of the observations.
     - Euclidean distance is used to measure the similarity among the observations.



  Sagemaker K-Means algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html
    Learning Type:
      - clustering (unsupervised)
    File/Data Types:
      -  Both recordIO-wrapped-protobuf and CSV formats are supported for training.
      - You can use either File mode or Pipe mode to train models on data that is formatted as
        recordIO-wrapped-protobuf or as CSV.
      - For inference, text/csv, application/json, and application/x-recordio-protobuf are supported.
        - k-means returns a closest_cluster label and the distance_to_cluster for each observation.
    Instance Type:
      - For training jobs, we recommend using CPU instances.
      - You can train on GPU instances, but should limit GPU training to single-GPU instances (such as
        ml.g4dn.xlarge) because only one GPU is used per instance

      - For inference, we recommend using CPU instances
    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/k-means-api-config.html
      required hyperparameters
        feature_dim
          - The number of features in the input data.
          - Valid values: Positive integer
        k
          - The number of required clusters.
          - Valid values: Positive integer
    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/k-means-tuning.html
      - The k-means algorithm computes the following metrics during training.
      - When tuning a model, choose one of these metrics as the objective metric.
      test:msd
         - Mean squared distances (MSD) between each record in the test set and the closest center of the model.
         - Minimize
      test:ssd
        - Sum of the squared distances (SSD) between each record in the test set and the closest center of the model.
        - Minimize


  K-Means Business use cases
    e-commerce and retail
      - to group customers based on their purchasing behavior and demographics.
    market segmentation
      - to group markets based on customer characteristics like age, income, interests, and buying habits.
    recommendation systems
      - to recommend products and services based on customer preferences.


  K-Means Algorithm
    https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html#km-instances

    K-Means Sample Notebooks
      - For a sample notebook that uses the SageMaker AI K-means algorithm to segment the population of counties
        in the United States by attributes identified using principle component analysis
        https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_applying_machine_learning/US-census_population_segmentation_PCA_Kmeans/sagemaker-countycensusclustering.html

         Saved to: notebooks/kmeans_notebooks/sagemaker-countycensusclustering.ipynb


