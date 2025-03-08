------------------------------------------------------
3.5 Examining SageMaker's Factorization Machines Algorithm

  SageMaker's Factorization Machines (FM) Algorithm.
    - It is a supervised machine learning algorithm that is an extension of a linear model designed to capture
      the higher order relationships between features in a dataset.

  higher-order vs Non-Linear Relationship
    higher-order
      - refers to interactions between three or more features in a dataset,
    non-linear
      - refers to a relationship between features that cannot be expressed by a linear function.
      - Since it involves two or more features and non-linear, the relationships is not necessarily higher order.

  factorization machine algorithm Equation
    simple linear regression equation:
       Y =  b + ∑ (m_i * x_i)     where ∑ (SUM) is from i=1 to i=n

            Y -> dependent feature or the feature we want to predict.
            b is a global bias.
            m_i represents the weight of the ith independent feature
            x_i -> one-hot encoded features

    pre-processing step
      - must convert all the categorical features into one-hot encoded features.

    equation to capture the higher order interaction.
       Y =  b + ∑_i (m_i * x_i) + ∑_i ∑_j <v_i dot v_j> x_i dot x_j
                where ∑ (SUM) is from i=1 to i=n

            v_i, v_j ->  latent vector representations of a feature.

       - The higher order is captured by the sum of sum of multiplying each one hot encoded columns
          alongside the dot product between the latent vector representation.

   Recommendation System Course
     - help understand factorization Machine Algorithm:
        Content-Based Recommendation Systems on Google Cloud
        - offered by Google Cloud
        - focus on 3rd module

  latent vector example
    - Considering a movie recommendation system where users rate movies on a scale of 1 to 5,
    - each movie is associated with certain features like genre, actors and directors.
    - Representing all these hidden features in a numerical fashion forms a latent vector.

  FM limitations with these algorithm
   considers only pairwise features.
     - In other words, it's only going to analyze the relationship between two pairs of features at a time,
       and that could be limiting depending on what you are trying to do with your problem.
   Does NOT support CSV support
     - does not support CSV,
   Does not support multi-class classification.
     - It only works for either binary classification or regression problems.
   Requires lots of data
     - to make up for the sparseness of the data, in other words, the missing features, it really needs
       a lot of data, and AWS recommends anywhere between 10,000 to 10 million rows in the dataset
   recommends CPUs only
     - use CPUs to give us the most efficient experience.
   does not perform well on dense data.


  Sagemaker Factorization Machine algorithm Attributes
    Learning Type:
      - Binary classification and regression
    File/Data Types:
      -  protobuf (with flow 32 tensors) and JSON (inference only)
    Instance Type:
      - CPU training and inference processes.
    Hyperparameters
      - required: feature_dim, num_factors, & predictor_type
      feature_dim,
        - determines the dimension of the input feature space.
      num_factors
        - determine the dimensionality of factorization
      predictor type,
        - determines the type of predictor.
    Metrics:
      - RMSE (regression metrics)
      - accuracy and cross-entropy (classification metrics)

  Factorization Machines Hyperparameters
    https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines-hyperparameters.html

    feature_dim
      - The dimension of the input feature space. This could be very high with sparse input.
      - Required
      - Valid values: Positive integer. Suggested value range: [10000,10000000]

    num_factors
     - The dimensionality of factorization.
     - Required
     - Valid values: Positive integer. Suggested value range: [2,1000], 64 typically generates good
       outcomes and is a good starting point.
     -  [dimension of matrix used algorithm calculation]

    predictor_type
      - The type of predictor.
      - binary_classifier: For binary classification tasks.
        regressor: For regression tasks.
       - Required
       - Valid values: String: binary_classifier or regressor

  Factorization machines Business Use Cases:
    recommendation systems and  ad-click prediction
       - where the data set is high dimensional and the data is sparse.


      Factorization Machines Algorithm
        https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html

        An Introduction to Factorization Machines with MNIST
        https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/factorization_machines_mnist/factorization_machines_mnist.html
           Saved to: notebooks/factorization_machines_notebooks/factorization_machines_mnist.ipynb


