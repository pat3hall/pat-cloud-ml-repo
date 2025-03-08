------------------------------------------------------
3.11 Discovering SageMaker's Object2Vec Algorithm `


  Object2Vec algorithm Overview
    - It is a highly customizable neural embedding algorithm that can be used to create vector representations of objects.
    - The objects can be anything such as words, sentences, or abstract entities like users or products.

  Object2Vec algorithm analogy
    - Imagine you are a librarian and your task is to organize a large number of books so that the readers
      can easily find them.
    - Each book has many attributes like genre, author, publication year, and so on.
    - Similarly, the readers have favorite genre, favorite authors, and preferences based on past reading habits.

    - The books are like the objects that this algorithm deals with.  Initially, the books may be placed in
      random order, but eventually you decide to create groups of books that are similar in any of the choose and attribute.
    - You mentally map out where each book should go, so similar books are placed together on the shelves.
    - This mental map is similar to creating an embedding space to organize the objects.
    - As readers start borrowing books, you start observing a pattern that some readers often borrow books from
      specific clusters, and based on this observation, you update the mental map, which makes it even easier for them.
    - This is like iterative adjustment of embeddings to better reflect the relationship between the objects.

  SageMaker Object2Vec algorithm analogy
    - During the training process, the algorithm accepts pairs of objects and the relationship labels as inputs.
      - For example, in a typical recommendation system, these pairs could be user and item.
      - These pairs are associated with labels indicating the nature of the relationships, whether the user
        liked or disliked the item.
    - Each object is initially presented as a random vector.
    - The goal is to adjust these vectors such that the objects with similar relationships are closer together.
    - Object2Vec uses a neural network to understand and learn embeddings.
    - For each object pair the neural network process their embeddings and predicts their relationships.
    - The predictor relationship is compared to the actual relationship using the loss function.
    - The error is then propagated back through the network, and the embeddings are updated to minimize the loss.
    - This process iteratively refines the embeddings to better capture the relationships between the objects.


  Sagemaker Object2Vec algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/object2vec.html
    Learning Type:
      - general-purpose neural embedding algorithm
    File/Data Types:
      - This algorithm expects the data to be provided in a sentence -sentence pair, label-sentence pair.
        and other pairs
          Sentence-sentence pairs
	    "A soccer game with multiple males playing." and "Some men are playing a sport."
          Labels- sequence pairs
	    The genre tags of the movie "Titanic", such as "Romance" and "Drama", and its short description:
            "James Cameron's Titanic is an epic, action-packed romance set against the ... of April 15, 1912."
          Customer-customer pairs
            The customer ID of Jane and customer ID of Jackie.
          Product-product pairs
            The product ID of football and product ID of basketball.

      - the data must be pre-processed and transformed into supported formats.
      - For training, the data must be in jsonlines format,
      - for inference, the data format must be in JSON or jsonlines format.
    Instance Type:
      - Amazon recommends CPU and GPU instances for training purposes.

    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/k-means-api-config.html
      required hyperparameters
        enc0_max_seq_len
          - The maximum sequence length for the enc0 encoder.
          - Valid values: 1 ≤ integer ≤ 5000
        enc0_vocab_size
          - The vocabulary size of enc0 tokens.
          - Valid values: 2 ≤ integer ≤ 3000000
    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/object2vec-tuning.html
      - This algorithm reports means square error (mse) for any regression tasks.
      - For classification tasks, it reports accuracy and cross entropy.

  Object2Vec Business use cases
     user behavior analysis
       - for creating detailed user profiles based on their interactions and behavior, which can be used
         for personalized marketing.
     natural language processing (NLP)
        - to detect spam and perform sentiment analysis.
     social network analysis
        - to identify groups of users with similar interests or behavior, which can be used for targeted advertising.



  An Introduction to SageMaker ObjectToVec model for sequence-sequence embedding
    https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/object2vec_sentence_similarity/object2vec_sentence_similarity.html
     Saved to: notebooks/object2vec_notebooks/object2vec_sentence_similarity.ipynb.txt


