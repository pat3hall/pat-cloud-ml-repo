------------------------------------------------------
3.13 Exploring SageMaker's Neural Topic Model (NTM) Algorithm


  Neural Topic Model (NTM) algorithm Overview
    - an unsupervised learning algorithm that is used to organize a corpus of documents into topics that
      contain word groupings based on their statistical distribution.

    - Similar to LDA, NTM is extensively used in the field of topic modeling.
    - Behind the scenes, NTM uses neural networks to capture complex patterns and provide a more nuanced
      understanding of the topics.
  LDA vs NTM Algorithms
    LDA
      modeling:
        - probabilistic graphical model using Dirichlet distributions
      scalability:
        - single instance CPU - as a result, it can be computationally expensive for large data sets
      interpretability:
        - highly interpretable due to its explicit probabilistic model
    NTM
      modeling:
        - uses neural network model.
      scalability:
        - can leverage both CPU and GPU instances and is very effective in handling large data sets.
      interpretability:
        - less interpretable because of the black box nature of the neural networks.



  Sagemaker NTM algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/ntm.html
    Learning Type:
      - unsupervised learning algorithm that specializes in text processing.
      - unsupervised learning algorithm that is used to organize a corpus of documents into topics that
        contain word groupings based on their statistical distribution
    File/Data Types:
     - For training, test, validate: CSV (dense only; file and pip mode) and protobuf (sparse and dense;
       file and pip mode) format
     - For inference: CSV, protobuf, JSON and JSONLines format
    Instance Type:
      - training supports both GPU and CPU instance types.
      - recommend GPU instances, but for certain workloads, CPU instances may result in lower training costs.
      - CPU instances should be sufficient for inference
    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/ntm_hyperparameters.html
      required hyperparameters
       num_topics
         - The number of required topics
         - Valid values: positive integer (min: 1, max: 1,000,000)
       feature_dim
         - The vocabulary size of the dataset
         - Valid values: positive integer (min: 1, max: 1,000,000)
    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/ntm-tuning.html
      validation:total_loss
        - Total Loss on validation set
        - goal: Minimize

  NTM Business use cases
    customer feedback analysis
      - to uncover customer pain points and suggest product improvements.
    personalized recommendations
      - to recommend articles, blog posts, and other contents based on the topic that a user has shown interest
    market sentiment analysis
      - to extract topics from financial news and social media
      - to gauge market sentiment and inform trading strategies.

   Note: Missing:
      Finally, in the resources section, you will see a page containing a sample notebook demonstrating
      SageMaker's NTM algorithm.

      Introduction to Basic Functionality of NTM
        https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/ntm_synthetic/ntm_synthetic.html

          Saved to: notebooks/ntm_notebooks/ntm_synthetic.ipynb.txt


