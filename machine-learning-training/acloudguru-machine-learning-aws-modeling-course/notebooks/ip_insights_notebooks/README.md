------------------------------------------------------
3.9 Understanding SageMaker's IP Insights Algorithm


  SageMaker's IP Insights algorithm
    - It's an unsupervised learning algorithm that learns the usage patterns for IPv4 addresses, capturing
      associations between IP addresses and entities like user IDs and account numbers.

  IP Insights example
    - Imagine you shop online on your favorite e-commerce platform from your home regularly.
    - Typically, you're shopping is on the weekend evenings.
    - You decided to go on a vacation to an international country.  On your vacation, you decided to do
      some shopping, but when you logged in, you are alerted with additional security questions and an
      OTP to your mobile phone to validate your authenticity.
      OTP: One time password or one-time pin
    - This is IP Insights algorithm in action,

  IP Insights algorithm
    - stores a usual IP address and the entity details like user ID or account number information as key value pairs.
    - Queries historical data for any login attempts and returns a score
    - A high score indicates an anomalous behavior
    - Uses a neural network

  Sagemaker IP Insight algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights.html
    Learning Type:
      - Anomaly detection (unsupervised)
    File/Data Types:
      - Training and validation data content types need to be in text/csv format.
      - The first column of the CSV data is an opaque string that provides a unique identifier for the entity.
      - The second column is an IPv4 address in decimal-dot notation.
        example CSV format:
           entity_id_1, 192.168.1.2
           entity_id_2, 10.10.1.2
      - supports only File mode.
      - For inference, IP Insights supports text/csv, application/json, and application/jsonlines data
    Instance Type:
      - run on both GPU and CPU
      - For training jobs, we recommend using GPU instances.
      - For inference, we recommend using CPU instances
    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights-hyperparameters.html
      required hyperparameters
        num_entity_vectors
          - The number of entity vector representations (entity embedding vectors) to train.
        vector_dim
          - represents the size of the key value pairs representing the entities and IP addresses.
          - The size of embedding vectors to represent entities and IP addresses.
          - The larger the value, the more information that can be encoded using these representations.
    Metrics
      -  It uses the optional validation channel to compute an area-under-curve (AUC) score on a predefined
         negative sampling strategy.
      - The AUC metric validates how well the model discriminates between positive and negative samples.

   JSONLines:
     - Each line of a JSONL file is a self-contained JSON object, like a mini database record.
num_entity_vectors

  IP Insights Business use cases
    Anomaly detection
      - fraud detectioin
         - fraudulent transactions and account takeovers.
      - compliance detection
          - compliance with regional regulations by detecting access from IP addresses located in
            restricted or high-risk regions.
    Geolocation based personalization
      - helps provide localized content and services based on the location of the IP addresses accessing the platform.

-----
An Introduction to the Amazon SageMaker IP Insights Algorithm
  https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/ipinsights_login/ipinsights-tutorial.html

[IPInsight] Model Hyperparameters
    num_entity_vectors:
      - the total number of embeddings to train.
      - use an internal hashing mechanism to map the entity ID strings to an embedding index; therefore, using
        an embedding size larger than the total number of possible values helps reduce the number of hash collisions.
      - recommend this value to be 2x the total number of unique entites (i.e. user names) in your dataset;

    vector_dim:
      - the size of the entity and IP embedding vectors.
      - The larger the value, the more information can be encoded using these representations but using too
        large vector representations may cause the model to overfit, especially for small training data sets;

-----


  IP Insights
    https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights.html

      IP Insights Sample Notebooks

       An Introduction to the Amazon SageMaker IP Insights Algorithm
       https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/ipinsights_login/ipinsights-tutorial.html
          saved to: notebooks/ip_insights_notebooks/ipinsights-tutorial.ipynb

