------------------------------------------------------
3.14 Understanding SageMaker's BlazingText Algorithm


   SageMaker's BlazingText algorithm overview
     - It is a highly optimized implementation of the word2vec and text classification algorithms.

   SageMaker's BlazingText algorithm
     - algorithm based on Facebook's FastText,
         - BlazingText is greatly scaled beyond FastText's performance.
           - where FastText may have taken days to train models, BlazingText can do that in minutes
           - inferences performed in real time, whereas FastText inference could only perform in batch mode.

     - BlazingText expects a single pre-processed text file.
        - Each line of the file should contain a single sentence,
        - if you need to train on multiple text files, just concatenate them all together, and upload as 1 file

      - AWS improved the scalability of the original word2vec algorithm, as well as the original, Facebook's
        FastText text classifier
           - BlazingText is 20 times faster than Facebook's FastText algorithm

   SageMaker's BlazingText algorithm mode

       Modes 	                Word2Vec                Text Classification
                                (Unsupervised           (Supervised
                                Learning)               Learning)
     -------------------        ----------------        ------------------

     Single CPU instance        cbow, Skip-gram          supervised
                                Batch Skip-gram

      Single GPU instance       cbow                     supervised with one GPU
      (with 1 or more GPUs)     Skip-gram

      Multiple CPU instances    Batch Skip-gram 	 None


-----
 skip-gram
   -"Skip-Gram" predicts the surrounding context words based on a single target word
   - essentially, Skip-Gram focuses on learning word representations by predicting context from a single word
 CBOW (Continuous Bag-of-Words)
   - CBOW predicts a target word based on its surrounding context words;
   - CBOW learns by predicting a word from its surrounding context
 skip-gram vs CBOW
   - Skip-Gram generally better for rare words and CBOW faster for large datasets with frequent words
-----

   SageMaker's BlazingText algorithm mode
     Word2Vec algorithm (mode = batch_skipgram, skipgram, or cbow)
       - The Word2vec algorithm is useful for many downstream natural language processing (NLP) tasks, such as
         sentiment analysis, named entity recognition, machine translation, etc
       - maps words to high-quality distributed vectors.
       - The resulting vector representation of a word is called a word embedding.
       - Words that are semantically similar correspond to vectors that are close together. That way, word
         embeddings capture the semantic relationships between word
       - provides the Skip-gram and continuous bag-of-words (CBOW) training architecture
       - BlazingText can generate meaningful vectors for out-of-vocabulary (OOV) words by representing their
         vectors as the sum of the character n-gram (subword) vectors.
     TextClassification algorithm (mode=supervised)
       - Text classification is an important task for applications that perform web searches, information retrieval,
         ranking, and document classification.
       - Ability to perform high speed multi-class and and multi-label text classification.
       - The goal of text classification is to automatically classify the text documents into one or more defined
         categories, like spam detection, sentiment analysis, or user reviews categorization.


  Sagemaker BlazingText algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html
    Learning Type:
      - unsupervised word2vec algorithm
      - supervised text classification algorithm
    File/Data Types:
      - expects a single pre-processed text file with space-separated tokens during the training, test, and validation stages.
      - JSON and jsonlines are supported during the inference stage.
    Instance Type:
      - single CPU and GPU instances for cbow and skip-gram modes, for Batch skip-gram,
      - support single or multiple CPU instances  (see previous table)

    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext_hyperparameters.html
      required hyperparameters
        mode
          - for word2vec architecture, valid values are: batch_skipgram, skipgram, or cbow
          - for text classification architecture, valid value: supervised
    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext-tuning.html
      Word2Vec algorithm reports
        train:mean_rho
          - The mean rho (Spearman's rank correlation coefficient) on WS-353 word similarity datasets
          - goal: Maximize
      text classification algorithm reports
        validation:accuracy
          - The classification accuracy on the user-specified validation dataset
          - goal: Maximize

  BlazingText Business use cases
    Word2Vec:
       - vectorize text (converted into real-valued vectors) for downstream natural language processing (NLP) tasks
         like sentiment analysis, named entity recognition, and machine translation
    Text Classification
       sentiment analysis
         - to evaluate customer reviews on a social media and decide if it is a positive or a negative sentiment.
       classify documents.
          - to crawl a series of documents in our enterprise and call out certain documents that may contain
            some sensitive information and be able to set those aside or give those documents extra protection.
       recommendation systems,
          - in recommending products, articles, or content based on user preferences.


    BlazingText Sample Notebooks
      For a sample notebook that trains and deploys the SageMaker AI BlazingText algorithm to generate word vectors,
      see Learning Word2Vec Word Representations using BlazingTex

        Learning Word2Vec Word Representations using BlazingText
           https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/blazingtext_word2vec_text8/blazingtext_word2vec_text8.html
           Saved to: notebooks/blazingtext_notebooks/blazingtext_word2vec_text8.ipynb



