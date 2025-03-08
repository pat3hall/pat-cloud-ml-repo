------------------------------------------------------
3.7 Reviewing SageMaker's Principal Component Analysis (PCA) Algorithm


  Principal Component Analysis (PCA) algorithm overview
    - It is an unsupervised learning algorithm that reduces the number of features in a dataset without
      losing meaningful information.
    - This algorithm helps overcome the curse of dimensionality problem.

  Principal Component Analysis (PCA) algorithm
    - combines uncorrelated, original features into components.
      - In other words, they are new features that are linear combinations of the original features.
    - The first component captures the maximum variance in the data,
    - The second component captures the remaining maximum variance orthogonal to the first component.
    - Eigenvalues and Eigenvectors are used to determine the magnitude and the direction of these components.

  PCA analogy
    - trying to photograph a 3D object that represents high dimensional data because the object has depth,
      width, height, color, and so on.
    - We take multiple pictures of this object at the best possible angles to capture all the important
      features in a two dimensional representation.
    - These best angles are analogous to the principal components trying to capture the most important information.


  PCA Modes of Operation
     Regular mode
       - specific for data sets with sparse data and a moderate number of features.
     Randomized.
       - For datasets with both a large number of observations and features.
       - This mode uses an approximation algorithm.


  Sagemaker PCA algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html
    Learning Type:
      - Dimensionality Reduction (unsupervised)
    File/Data Types:
      - training dataset: CSV and protobuf in File mode or Pipe mode
      - or inference, PCA supports text/csv, application/json, and application/x-recordio-protobuf
    Instance Type:
      - PCA supports CPU and GPU instances for training and inference (depends on training data)
    Hyperparameters
      - required: feature_dim, mini_batch_size, num_components
      feature_dim
        - that specifies the input dimension.
      mini_batch_size
        - the number of rows in a mini batch
      num_components
        - the number of principal components to compute.

  PCA hyperparameters:
    https://docs.aws.amazon.com/sagemaker/latest/dg/PCA-reference.html
    feature_dim
      - Input dimension.
       - Required
       - Valid values: positive integer
    mini_batch_size
      - Number of rows in a mini-batch.
      - Required
      - Valid values: positive integer
    num_components
      - The number of principal components to compute.
      - Required
      - Valid values: positive integer

  PCA Business Use Cases:
    image compression,
       - reducing the pixel data by preserving the important features.
    financial analysis.
       - By reducing the number of variables, PCA helps in identifying the factors driving market movements.
    Customer feedback analysis.
       - By reducing the dimensionality of the data, PCA can help identify the themes and sentiments provided
         in the user feedback.


  Principal Component Analysis (PCA) Algorithm
    https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html

       PCA Sample Notebooks
        https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/pca_mnist/pca_mnist.html
          -> save to notebooks/pca_notebooks/pca_minst.ipynb


