------------------------------------------------------
3.12 Reviewing SageMaker's Latent Dirichlet Allocation (LDA) Algorithm


  Latent Dirichlet allocation (LDA) Algorithm Overview
    - an unsupervised learning algorithm that attempts to describe a set of observations as a mixture
      of different categories.
    - Each observation is considered a document.
    - The features are the presence of each word and the categories are the topics.
    - a generative probabilistic model used for discovering the underlying topics in a collection of documents.
    - extensively used in the field of natural language processing for topic modeling purposes.

  LDA Algorithm - an Analogy
             represent
      books     ->    Documents in dataset
      Genre     ->    Hidden Topic
      words     ->    words in the Documents


  Sagemaker LDA algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/lda.html
    Learning Type:
      - Unsupervised learning commonly used for topic modeling (to discover a user-specified number of
        topics shared by documents within a text corpus.
    File/Data Types:
     - For training: CSV (dense only; file mode) and protobuf (sparse and dense; file and pip mode) format
     - For inference: CSV, protobuf, and JSON format
     - JSON is also supported during the inference stage.
    Instance Type:
      - supports single-instance CPU training.
      - CPU instances are recommended for hosting/inference.
    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/k-means-api-config.html
      required hyperparameters
       num_topics
         - The number of topics for LDA to find within the data.
         - Valid values: positive integer
       feature_dim
         - The size of the vocabulary of the input document corpus.
         - Valid values: positive integer
       mini_batch_size
         - The total number of documents in the input document corpus.
         - Valid values: positive integer
    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/lda-tuning.html
      test:pwll
        - Per-word log-likelihood on the test dataset.
        - The likelihood that the test dataset is accurately described by the learned LDA model.
        - goal: Maximize

  LDA Business use cases
    customer feedback analysis
      - to identify the common themes and topics mentioned in the customer reviews.
    social media trend analysis
      - to identify trending topics on social media and understand public sentiment and emerging issues.
    content creation
      - to generate ideas from blog posts, articles, and marketing content based on the topics that are
        of interest to the target audience.


    An Introduction to SageMaker LDA
      https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/lda_topic_modeling/LDA-Introduction.html

        Saved to:  notebooks/lda_notebooks/LDA-Introduction.ipynb

