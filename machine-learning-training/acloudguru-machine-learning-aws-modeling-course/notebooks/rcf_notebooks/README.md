------------------------------------------------------
3.8 Exploring SageMaker's Random Cut Forest (RCF) Algorithm


  SageMaker's Random Cut Forest (RCF) algorithm.
    - It is an unsupervised algorithm primarily used to detect anomalies in a dataset.

  anomaly, or an outlier
    - is an extreme value that deviates significantly from most data points.

  Random Cut Forest (RCF)
    - The algorithm computes the distance from the mean for every data point and assigns a score
      to each of them
      - A low score that is a score less than three standard deviations from the mean score indicates
        a normal data point.
      - a high score indicates an anomaly.
    - Reservoir sampling is a common algorithm often used to effectively draw random samples from a dataset.



  Sagemaker RCF algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html
    Learning Type:
      - Anomaly detection
    File/Data Types:
      - training dataset: CSV and protobuf in File mode or Pipe mode
      - Train and test data content types can be either application/x-recordio-protobuf or text/csv formats.
      - For the test data, when using text/csv format, the content must be specified as text/csv;label_size=1
        where the first column of each row represents the anomaly label: "1" for an anomalous data point and "0"
        for a normal data point.
      - You can use either File mode or Pipe mod
      - For inference, RCF supports application/x-recordio-protobuf, text/csv and application/json input dat
    Instance Type:
      - recommends only CPU instances (does not take advantage of GPU hardware)
    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/rcf_hyperparameters.html
      - required: feature_dim
      feature_dim
        - The number of features in the data set.
        - (If you use the Random Cut Forest estimator, this value is calculated for you and need not be specified.)
        - Required
        - Valid values: Positive integer (min: 1, max: 10000)
    Metrics
      -  The optional test channel is used to compute accuracy, precision, recall, and F1-score metrics on labeled data.


 Random Cut Forest (RCF) business use cases
   anomaly detection.
     fraud detection
       - by detecting unusual patents in financial transaction data,
     Security breaches
       - detecting security breaches by analyzing network traffic that could signify security breaches
         or cyber attacks.
   e-commerce
     - to detect unusual shopping patterns that might indicate bot activities or fraudulent purchases.


  Random Cut Forest (RCF) Algorithm
    https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html
      RCF Sample Notebooks
        https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/random_cut_forest/random_cut_forest.html
        Saved to: notebooks/rcf_notebooks/random_cut_forest.ipynb


