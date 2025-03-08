------------------------------------------------------
4.6 Using Apache Spark with Amazon SageMaker


  Saved files:
    completed python jupyter notebook:
      pyspark_mnist_xgboost.ipynb
    extracted python code from jupyter notebook:
      pyspark_mnist_xgboost.py


   Apache Spark
     - an open source distributed computing system designed for fast processing of large scale data by distributing
       the computational tasks across multiple nodes in a cluster.

   Training a Model Using Apache Spark
      - use the MNIST database, which is a large database of handwritten digits as the input data.
      - In the setup process:
          - from SageMaker Notebook, we will create a Spark session,
          - create a Spark data frame, and load the input data into this data frame.
          -  split data into training and validation data set.
       - use XGBoost SageMaker Estimator


   SageMaker Notebook instance -> Jupyter -> SageMaker Examples -> SageMaker Spark -> pyspark_minst_xgboost.ipynb -> Use

   You can visit SageMaker Spark's GitHub repository at
      https://github.com/aws/sagemaker-spark to learn more about SageMaker Spark.

   You can visit XGBoost's GitHub repository at
     https://github.com/dmlc/xgboost to learn more about XGBoost

  LibSVM format:
    - The first row contains the class label, in this case 0 or 1.
    - Following that are the features, here there are two values for each one; the first one is the feature
      index (i.e. which feature it is) and the second one is the actual value.
    - The feature indices starts from 1 (there is no index 0) and are in ascending order.
    - The indices not present on a row are 0.
    - In summary, each row looks like this;
      <label> <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>
    - This format is advantageous to use when the data is sparse and contain lots of zeroes.
    - All 0 values are not saved which will make the files both smaller and easier to read.

  What is Spark’s JAR Folder?
    https://medium.com/@Nelsonalfonso/understanding-sparks-jar-folder-its-location-and-usage-7817db37cb27
    - The Spark JAR folder is the repository of library files that Spark uses during its operations.
    - These library files or JAR files contain compiled Java classes and associated metadata that encapsulate
      the core functionalities of Spark.
    - The location of the Spark JAR folder varies depending on the Spark installation method and the
      operating system in use.



SageMaker PySpark Example notebooks:

  SageMaker Notebook instance -> Jupyter -> SageMaker Examples -> SageMaker Spark ->

    pyspark_mnist_custom_estimator.ipynb
    pyspark_mnist_kmeans.ipynb
    pyspark_mnist_pca_kmeans.ipynb
    pyspark_mnist_pca_mllib_kmeans.ipynb
    pyspark_mnist_xgboost.ipynb

