Amazon SageMaker’s Built-in Algorithm Webinar Series: Blazing Text
Amazon Web Services
    
Summary:
  In this webinar which covers the Blazing Text algorithm used by Amazon SageMaker - 
  https://amzn.to/2S1lZWD, Pratap Ramamurthy, AWS Partner Solution Architect, 
  will show you how to use Blazing Text for classification, natural language generation and sentiment analysis on text. 


   Amazon SageMaker’s Built-in Algorithm Webinar Series:
     1  Image Classification
     2  Clustering
     3  Sequence2Sequence
     4  Linear Learner
     5  DeepAR
     6  Blazing Text
     7  OCR
     8  XGBoost
     9  LDA
     10 Factorization Machines
     11 Randon Cut Forest
     13 Neural Topic Model


  What are Amazon Algorithm's
    - popular algorithms, complete written by AWS, optimized, consumed as a Service, available through Sagemaker

  Blazing Text Algorithm use cases
    - text classifiers
    - natural language generators
    - sentiment Analysis
    - Any Natural Language Processing (NLP) task
      - i.e. where you need to vectorize your words


  Typical Deep Learning task on Task

     example: 3 layer neural network
       --->  input layer ---> hidden layer 1 ---> hidden layer 2  ---> output layer

         - neurons take a number, typically 0 to 1 or -1 to 1, as input
         - to send text to a neural network, the words need to be converted to digital representation
           before being consumed by a neural network. This is called vectorization or embedding

     Types of Encoding
       integer encoding
         - assign an integer to each word (e.g. if in alpabetical order: a -> 1, able -> 2, ...)
         - useful for somethings, but not for natural language processing tasks
            - example your trying to be predict 'acid' which is '4', and your slightly off, you may
              get '3' which is 'account' (unrelated words)
         - for NLP, words close together need to have some type of similarity
       one-hot encoding
           a       -> [1 0 0 0 0 ...] 250,000 dimensions
           able    -> [0 1 0 0 0 ...] 
           account -> [0 0 1 0 0 ...] 
           acid    -> [0 0 0 1 0 ...] 
           ...        ....
           zone    -> [0 0 0 0 0 ...  0 0 0 1] 

            - all words are equally dissimilar for each other, that is, all words are equidistance
              from each other
            - the problem here is too many zeros resulting in very large vectors, requiring alot of compute memory 
            - between 50k and 400K words in English language

       dense (or sparse) representation
          - example, reduce 250K dimensions to 250 dimension and have a nonzero number in each dimension

     Requirements for word vectors in an Embedding
       - solved in seminal paper called 'Algorithms Words2Vec (words to vectors)'
       Meaningful representation
         - words with similar context shoud be clustered together
         - distance between similar words should be smaller than words with opposite meanings
       Dense matrix (or dense representation)
         - don't want alot of zeros because that wastes memory
       Unsupervised
         - no human intervention
       Intuitive


     Word2Vec mechanism

       Word2Vec setup

         start with a         Set               Unsupervised         Generate
         corpus of     ---->  parameters ---->  learning      ---->  Dense 
         text                                                        embedding

            - most important parameter is the dimensions (what is the dimensionality of the vectors)
              (e.g. 250, 300, ...)

       Skip-gram Preprocessing step

         from:  'Word2Vec Tutorial - The Skip-Gram Model' by Chris McCormick 
           http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

         - convert a large amount of words and convert it into training examples
         - feeding diagrams or Ingrams or what is called skip-grams
         - skip-gram is a moving window to create word pairs that pair the currently selected workd (___)
           with other words in its vicinity (----), in this example, pairs words within 2 words of selected word 

         Source Text                                        Training Samples
         -------------------------------------------        ----------------
         The quick brown fox jumps over the lazy dog  --->   (the, quick)
         ___ ----- -----                                     (the, brown)

         The quick brown fox jumps over the lazy dog  --->   (quick, the)
         --- _____ ----- ----                                (quick, brown)
                                                             (quick, fox)

         The quick brown fox jumps over the lazy dog  --->   (brown, the)
         --- ----- _____ --- -----                           (brown, quick)
                                                             (brown, fox)
                                                             (brown, jumps)

         The quick brown fox jumps over the lazy dog  --->   (fox, quick)
             ----- ----- ___ ----- ----                      (fox, brown)
                                                             (fox, jumps)
                                                             (fox, jumps)



       Neural network setup
         - a very shallow neural network with an input layer, 1 hidden layer, and an output layer 
         - context word is at the output in one-hot form (e.g input:  quick; output context word: brown) 
         - output layer is equal to vocabulary size (one-hot size)
         - hidden layer size is set to diminsionality of your choice (e.g. 250, 300, ...)
         - once this neural network is trained by the skip-gram from the entire corpus, we throw away 
           this network, but save the weights of the hidden layer and use them to create a transformation table 
             - just need to representations of the words after the training

           input          hidden layer         Output layer
           Vector   ----> Linear Neurons ----> Softmax Classifier  ---> Probabiity that a word at a
                                                                        a randomly chosen, nearby 
          one-hot          300 Neurons          10k neurons             position is 'abandom' 
          vector
          10k positions

    
       BlazingText word embedding
          example: 5 five dimensional representation of words (note: requires much larger dimensionality than 5)
             - dense representation - almost every value is non-zero value
             - Word2vec algorithm converts words to vectors 
             - BlazingText is a Word2Vec algorithm with optimizations for AWS enviroment

                word       Encoding
                -------    -------------------------
                a          [.01 .83 .49 .01 .29]
                able       [.83 .83 .49 .01 .29]
                acccount   [.64 .92 .05 .38 .80]
                acid       [.85 .63 .03 .28 .37]
                            .  .  .
                zone       [.78 .54 .90 .23 .84]

       Word Vectors used for further ML Training
         - word vectors are inputs to other deep ML tasks  (e.g. sentimental tasks)

           word            input      hidden        hidden         Output
           Vectors  ---->  Layer ---> layer 1 ----> layer 2  ----> layer  

       Intuition behind word2vec
          - shows a t-SNE (t-distributed Stochasitic Neighbor Embedding) plot of words
          - plot shows a 300 dimensional vector reduced to two dimensions on output data from blazingText
            - plot shows many similar words are closely clustered together
          
          Notes: 
             sto·chas·tic
               - randomly determined; having a random probability distribution or pattern that may be 
                 analyzed statistically but may not be predicted precisely.
             t-distributed stochastic neighbor embedding (t-SNE) 
               - is a statistical method for visualizing high-dimensional data by giving each datapoint a 
                 location in a two or three-dimensional map. 


       How did the magic work?
         - context: the set of words that co-occur (close proximity)
         - how to take the weight of the hidden layer and use it appropriately?
         - first step of the neural network: send the one-hot encoded vector into a neural network and have
           a matrix multiplication

            one-hot         Weights      select weights (row) for the word 
            input 
                          | 17 24  1 |
                          | 23  5  7 |
           [0 0 0 1 0] x  |  4  6 13 |  = [10 12 19]
                          | 10 12 19 |
                          | 11 18 28 |



       OOV (out of vocubalary) Handling Using Blazing Text
          - handling new words that are not in your corpus used to training the weights (or spelling errors) 
         
         subword detection
           - instead of taking an entire word as one word, take subwords 
           - helps with some types of OOV words (similar word such as missing pural, composed of two+ words 
             included in corpus
           - may not help words that have no subwords in the training corpus

                           <ap
                           app
                           appl
                           apple
                           apple>
             apple  ---->  ppl       ---->  Vectorize
                           pple>
                           ple
                           ple>

       Text Classification with Blazing Text
         - AWS implementation of Word2Vec plus additional optimizations
         - BlazingText adds a text classifier
           - instead of just converting words to vectors, 

         Typical NLP Pipeline
            preprocessing ----> Transform to word vectors ----> Classifier

             - preprocessing step to convert the words into skip-gram (or other similar methods)
             - transform skip-gram inputs to word vectors (using BlazingText)
             - apply a classifier such as a deep neural network or LSTM (?) for your classification problem
             - blazingText provides Word2Vec combined with Classification

         Blazing Text Classifier Input:

            __label__4   linux ready for prime time, intel says, despite all the linux hype

            __label__2   bowled by slower one again, the past caught up with sourav ganguly 

            - provide a label with '__label__' prefix and then the text

         BlazingText Word2Vec Parameters:
           mode: 
             - The Word2vec architecture used for training.
             - Valid values: batch_skipgram, skipgram, or cbow

           Continuous Bag of Words (cbow)
             - opposite of skip-gram
             - provide the context words and expect it to predict the actual word
               
                     context             word
                     ----                -------
                     the
                     quick    --------->  brown
                     fox
                     jumps


           Skip-Gram
             - provide the words, and expect the neural network to predict the context
             - generally, skip-gram is preferred and it gives a slightly higher accuracy

                                                                    word   context
               The quick brown fox jumps over the lazy dog  --->   (brown, the)
               --- ----- _____ --- -----                           (brown, quick)
                                                                   (brown, fox)
                     word                  context                 (brown, jumps)
                     ----                  -------
                                           the
                                           quick
                     brown   ------------> fox
                                           jumps

             CPU vs GPU:
                   Note: does not use GPUs all the time due to some optimizations that need to converge quickly
                         - subword sampling, etc. are harder to run on a GPU

                               |-------------------------------------------------------------|----------------|
                      algorithm|                   word2Vec                                  | Text           |
                               |                                                             | Classification |
                               |-------------------------------------------------------------|----------------|
                       mode:   |Cbow (supports     | skipgram (supports     | batch_skipgram | supervised     |
                               |subword training)  |  subword training)     |                |                |
                               |-------------------|------------------------|----------------|----------------|
                    single CPU |       x           |          x             |     x          |      x         |
                    instance   |                   |                        |                |                |
                               |-------------------|------------------------|----------------|----------------|
                    single GPU |       x           |          x             |                |   x (instance  |
                    instance   |                   |                        |                |with 1 GPU only)|
                               |-------------------|------------------------|----------------|----------------|
                  multiple CPU |                   |                        |     x          |                |
                    instances  |                   |                        |                |                |
                               |-------------------|------------------------|----------------|----------------|




    Dataset for BlazingText Word2Vec demo:
      https://mattmahoney.net/dc/textdata.html
      - The test data for the Large Text Compression Benchmark is the first 10**9 bytes of the English Wikipedia dump on Mar. 3, 2006. 
        http://download.wikipedia.org/enwiki/20060303/enwiki-20060303-pages-articles.xml.bz2 (1.1 GB or 4.8 GB after decompressing with bzip2 - link 
        no longer works). Results are also given for the first 10**8 bytes, which is also used for the Hutter Prize. 

    BlazingText Word2Vec hyperparameters:

      mode="batch_skipgram",
        - The Word2vec architecture used for training.
        - Valid values: batch_skipgram, skipgram, or cbow

      epochs=5,
        - The number of complete passes through the training data.
        - Optional; Valid values: Positive integer; Default value: 5

      min_count=5,
        - Words that appear less than min_count times are discarded.
        - Optional; Valid values: Non-negative integer; Default value: 5

      sampling_threshold=0.0001,
        - The threshold for the occurrence of words. Words that appear with higher frequency in the training data are randomly down-sampled.
        - Optional; Valid values: Positive fraction. The recommended range is (0, 1e-3);  Default value: 0.0001
        
      learning_rate=0.05,
        - The step size used for parameter updates.
        - Optional; Valid values: Positive float; Default value: 0.05

      window_size=5,
        - The size of the context window. The context window is the number of words surrounding the target word used for training.
        - Optional; Valid values: Positive integer; Default value: 5

      vector_dim=100,
        - The dimension of the word vectors that the algorithm learns.
        - Optional; Valid values: Positive integer; Default value: 100

      negative_samples=5,
        - The number of negative samples for the negative sample sharing strategy.
        - Optional; Valid values: Positive integer; Default value: 5

      batch_size=11,  #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)

      evaluation=True,  # Perform similarity evaluation on WS-353 dataset at the end of training

      subwords=False,
       - Whether to learn subword embeddings on not.
       - Optional; Valid values: (Boolean) True or False; Default value: False
       - Subword embedding learning is not supported by batch_skipgram




    code: blazingtext_word2vec_text8.ipynb

        >>> # # Learning Word2Vec Word Representations using BlazingText
        >>> # 


        >>> # Word2Vec is a popular algorithm used for generating dense vector representations of words in large corpora using unsupervised learning. The resulting vectors have been shown to capture semantic relationships between the corresponding words and are used extensively for many downstream natural language processing (NLP) tasks like sentiment analysis, named entity recognition and machine translation.  

        >>> # SageMaker BlazingText which provides efficient implementations of Word2Vec on
        >>> # 
        >>> # - single CPU instance
        >>> # - single instance with multiple GPUs - P2 or P3 instances
        >>> # - multiple CPU instances (Distributed training)

        >>> # In this notebook, we demonstrate how BlazingText can be used for distributed training of word2vec using multiple CPU instances.

        >>> # ## Setup
        >>> # 
        >>> # Let's start by specifying:
        >>> # - The S3 buckets and prefixes that you want to use for saving model data and where training data is located. These should be within the same region as the Notebook Instance, training, and hosting. If you don't specify a bucket, SageMaker SDK will create a default bucket following a pre-defined naming convention in the same region. 
        >>> # - The IAM role ARN used to give SageMaker access to your data. It can be fetched using the **get_execution_role** method from sagemaker python SDK.

        >>> import sagemaker
        >>> from sagemaker import get_execution_role
        >>> import boto3
        >>> import json

        >>> sess = sagemaker.Session()

        >>> role = get_execution_role()
        >>> print(
        >>>     role
        >>> )  # This is the role that SageMaker would use to leverage AWS resources (S3, CloudWatch) on your behalf

        >>> region = boto3.Session().region_name

        >>> output_bucket = sess.default_bucket()  # Replace with your own bucket name if needed
        >>> print(output_bucket)
        >>> output_prefix = "sagemaker/DEMO-blazingtext-text8"  # Replace with the prefix under which you want to store the data if needed

        >>> data_bucket = (
        >>>     f"sagemaker-example-files-prod-{region}"  # Replace with the bucket where your data is located
        >>> )
        >>> data_prefix = "datasets/text/text8/text8"


        >>> # ### Data Ingestion
        >>> # 
        >>> # BlazingText expects a single preprocessed text file with space separated tokens and each line of the file should contain a single sentence. In this example, let us train the vectors on [text8](http://mattmahoney.net/dc/textdata.html) dataset (100 MB), which is a small (already preprocessed) version of Wikipedia dump. Data is already downloaded from [matt mahoney's website](http://mattmahoney.net/dc/text8.zip), uncompressed and stored in `data_bucket`. 

        >>> s3_client = boto3.client("s3")
        >>> s3_client.download_file(data_bucket, data_prefix, "text8")
        >>> s3_client.upload_file("text8", output_bucket, output_prefix + "/train")

        >>> s3_train_data = f"s3://{output_bucket}/{output_prefix}/train"


        >>> # Next we need to setup an output location at S3, where the model artifact will be dumped. These artifacts are also the output of the algorithm's training job.

        >>> s3_output_location = f"s3://{output_bucket}/{output_prefix}/output"


        >>> # ## Training Setup
        >>> # Now that we are done with all the setup that is needed, we are ready to train our object detector. To begin, let us create a ``sageMaker.estimator.Estimator`` object. This estimator will launch the training job.

        >>> region_name = boto3.Session().region_name


        >>> container = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "blazingtext", "latest")
        >>> print(f"Using SageMaker BlazingText container: {container} ({region_name})")


        >>> # ## Training the BlazingText model for generating word vectors

        >>> # Similar to the original implementation of [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf), SageMaker BlazingText provides 
        >>> # an efficient implementation of the continuous bag-of-words (CBOW) and skip-gram architectures using Negative Sampling, on CPUs 
        >>> # and additionally on GPU[s]. The GPU implementation uses highly optimized CUDA kernels. To learn more, please refer to [*BlazingText: 
        >>> # Scaling and Accelerating Word2Vec using Multiple GPUs*](https://dl.acm.org/citation.cfm?doid=3146347.3146354). BlazingText also 
        >>> # supports learning of subword embeddings with CBOW and skip-gram modes. This enables BlazingText to generate vectors for 
        >>> # out-of-vocabulary (OOV) words, as demonstrated in this [notebook]
        >>> # (https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/blazingtext_word2vec_subwords_text8/blazingtext_word2vec_subwords_text8.ipynb).
        >>> # 
        >>> # 
        >>> # 

        >>> # Besides skip-gram and CBOW, SageMaker BlazingText also supports the "Batch Skipgram" mode, which uses efficient mini-batching and 
        >>> # matrix-matrix operations ([BLAS Level 3 routines](https://software.intel.com/en-us/mkl-developer-reference-fortran-blas-level-3-routines)). 
        >>> # This mode enables distributed word2vec training across multiple CPU nodes, allowing almost linear scale up of word2vec computation to 
        >>> # process hundreds of millions of words per second. Please refer to [*Parallelizing Word2Vec in Shared and Distributed Memory*]
        >>> # (https://arxiv.org/pdf/1604.04661.pdf) to learn more.

        >>> # BlazingText also supports a *supervised* mode for text classification. It extends the FastText text classifier to leverage GPU 
        >>> # acceleration using custom CUDA kernels. The model can be trained on more than a billion words in a couple of minutes using a multi-core 
        >>> # CPU or a GPU, while achieving performance on par with the state-of-the-art deep learning text classification algorithms. For more 
        >>> # information, please refer to [algorithm documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html) or 
        >>> # [the text classification notebook]
        >>> # (https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia/blazingtext_text_classification_dbpedia.ipynb).

        >>> # To summarize, the following modes are supported by BlazingText on different types instances:
        >>> # 
        >>> # |          Modes         	| cbow (supports subwords training) 	| skipgram (supports subwords training) 	| batch_skipgram 	| supervised |
        >>> # |:----------------------:	|:----:	|:--------:	|:--------------:	| :--------------:	|
        >>> # |   Single CPU instance  	|   ✔  	|     ✔    	|        ✔       	|  ✔  |
        >>> # |   Single GPU instance  	|   ✔  	|     ✔    	|                	|  ✔ (Instance with 1 GPU only)  |
        >>> # | Multiple CPU instances 	|      	|          	|        ✔       	|     | |
        >>> # 
        >>> # Now, let's define the resource configuration and hyperparameters to train word vectors on *text8* dataset, using "batch_skipgram" mode on two c4.2xlarge instances.
        >>> # 

        >>> bt_model = sagemaker.estimator.Estimator(
        >>>     container,
        >>>     role,
        >>>     instance_count=2,
        >>>     instance_type="ml.c4.2xlarge",
        >>>     train_volume_size=5,
        >>>     train_max_run=360000,
        >>>     input_mode="File",
        >>>     output_path=s3_output_location,
        >>>     sagemaker_session=sess,
        >>> )


        >>> # Please refer to [algorithm documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext_hyperparameters.html) for the complete list of hyperparameters.

        >>> bt_model.set_hyperparameters(
        >>>     mode="batch_skipgram",
        >>>     epochs=5,
        >>>     min_count=5,
        >>>     sampling_threshold=0.0001,
        >>>     learning_rate=0.05,
        >>>     window_size=5,
        >>>     vector_dim=100,
        >>>     negative_samples=5,
        >>>     batch_size=11,  #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
        >>>     evaluation=True,  # Perform similarity evaluation on WS-353 dataset at the end of training
        >>>     subwords=False,
        >>> )  # Subword embedding learning is not supported by batch_skipgram


        >>> # Now that the hyper-parameters are setup, let us prepare the handshake between our data channels and the algorithm. 
        >>> # To do this, we need to create the `sagemaker.session.s3_input` objects from our data channels. These objects 
        >>> # are then put in a simple dictionary, which the algorithm consumes.

        >>> train_data = sagemaker.session.s3_input(
        >>>     s3_train_data,
        >>>     distribution="FullyReplicated",
        >>>     content_type="text/plain",
        >>>     s3_data_type="S3Prefix",
        >>> )
        >>> data_channels = {"train": train_data}


        >>> # We have our `Estimator` object, we have set the hyper-parameters for this object and we have our data channels 
        >>> # linked with the algorithm. The only  remaining thing to do is to train the algorithm. The following command will 
        >>> # train the algorithm. Training the algorithm involves a few steps. Firstly, the instance that we requested while 
        >>> # creating the `Estimator` classes is provisioned and is setup with the appropriate libraries. Then, the data from 
        >>> # our channels are downloaded into the instance. Once this is done, the training job begins. The provisioning and 
        >>> # data downloading will take some time, depending on the size of the data. Therefore it might be a few minutes before 
        >>> # we start getting training logs for our training jobs. The data logs will also print out `Spearman's Rho` on 
        >>> # some pre-selected validation datasets after the training job has executed. This metric is a proxy for the quality of the algorithm. 
        >>> # 
        >>> # Once the job has finished a "Job complete" message will be printed. The trained model can be found in the S3 bucket 
        >>> # that was setup as `output_path` in the estimator.

        >>> bt_model.fit(inputs=data_channels, logs=True)
            INFO:sagemaker:Creating training-job with name: blazingtext-2024-11-04-23-39-41-133
            
            2024-11-04 23:39:46 Starting - Starting the training job...
            2024-11-04 23:39:59 Starting - Preparing the instances for training...
            2024-11-04 23:40:29 Downloading - Downloading input data...
            2024-11-04 23:41:00 Training - Training image download completed. Training in progress..Arguments: train
            . . .
            . . .
            Alpha: 0.0413  Progress: 17.60%  Million Words/sec: 5.82
            Alpha: 0.0387  Progress: 22.79%  Million Words/sec: 5.86
            Alpha: 0.0362  Progress: 27.97%  Million Words/sec: 5.77
            Alpha: 0.0337  Progress: 33.14%  Million Words/sec: 5.79
            Alpha: 0.0311  Progress: 38.42%  Million Words/sec: 5.79
            Alpha: 0.0285  Progress: 43.58%  Million Words/sec: 5.80
            Alpha: 0.0259  Progress: 48.75%  Million Words/sec: 5.67
            Alpha: 0.0234  Progress: 53.90%  Million Words/sec: 5.70
            Alpha: 0.0208  Progress: 58.96%  Million Words/sec: 5.69
            Alpha: 0.0183  Progress: 64.08%  Million Words/sec: 5.71
            Alpha: 0.0157  Progress: 69.16%  Million Words/sec: 5.71
            Alpha: 0.0132  Progress: 74.33%  Million Words/sec: 5.69
            Alpha: 0.0106  Progress: 79.41%  Million Words/sec: 5.70
            Alpha: 0.0080  Progress: 84.48%  Million Words/sec: 5.69
            Alpha: 0.0055  Progress: 89.56%  Million Words/sec: 5.70
            Alpha: 0.0027  Progress: 95.27%  Million Words/sec: 5.62
            Alpha: 0.0000  Progress: 100.00%  Million Words/sec: 5.46
            Training finished!
            Average throughput in Million words/sec: 5.46
            Total training time in seconds: 15.58
            Evaluating word embeddings....
            Vectors read from: /opt/ml/model/vectors.txt 
            {
                "EN-WS-353-ALL.txt": {
                    "not_found": 2,
                    "spearmans_rho": 0.7078008535992815,
                    "total_pairs": 353
                },
                "EN-WS-353-REL.txt": {
                    "not_found": 1,
                    "spearmans_rho": 0.6786414568405045,
                    "total_pairs": 252
                },
                "EN-WS-353-SIM.txt": {
                    "not_found": 1,
                    "spearmans_rho": 0.7426563438123545,
                    "total_pairs": 203
                },
                "mean_rho": 0.7096995514173802
            }
            [11/04/2024 23:41:49 INFO 140700728170304] #mean_rho: 0.7096995514173802
            [11/04/2024 23:41:51 INFO 139963175397184] Master host is not alive. Training might have finished. Shutting down.... Check the logs for algo-1 machine.
            
            2024-11-04 23:42:09 Uploading - Uploading generated training model
            2024-11-04 23:42:09 Completed - Training job completed
            Training seconds: 198
            Billable seconds: 198

        >>> # ## Hosting / Inference
        >>> # Once the training is done, we can deploy the trained model as an Amazon SageMaker real-time hosted endpoint. 
        >>> # This will allow us to make predictions (or inference) from the model. Note that we don't have to host on the same 
        >>> # type of instance that we used to train. Because instance endpoints will be up and running for long, it's advisable 
        >>> # to choose a cheaper instance for inference.

        >>> bt_endpoint = bt_model.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")


        >>> # ### Getting vector representations for words

        >>> # #### Use JSON format for inference
        >>> # The payload should contain a list of words with the key as "**instances**". BlazingText supports content-type `application/json`.

        >>> words = ["awesome", "blazing"]

        >>> payload = {"instances": words}

        >>> response = bt_endpoint.predict(
        >>>     json.dumps(payload),
        >>>     initial_args={"ContentType": "application/json", "Accept": "application/json"},
        >>> )

        >>> vecs = json.loads(response)
        >>> print(vecs)


        >>> # As expected, we get an n-dimensional vector (where n is vector_dim as specified in hyperparameters) for each of the words. If the word is not there in the training dataset, the model will return a vector of zeros.

        >>> # ### Evaluation

        >>> # Let us now download the word vectors learned by our model and visualize them using a [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) plot.

        >>> s3 = boto3.resource("s3")

        >>> key = bt_model.model_data[bt_model.model_data.find("/", 5) + 1 :]
        >>> s3.Bucket(output_bucket).download_file(key, "model.tar.gz")


        >>> # Uncompress `model.tar.gz` to get `vectors.txt`

        >>> get_ipython().system('tar -xvzf model.tar.gz')


        >>> # If you set "evaluation" as "true" in the hyperparameters, then "eval.json" will be there in the model artifacts.
        >>> # 
        >>> # The quality of trained model is evaluated on word similarity task. We use [WS-353](http://alfonseca.org/eng/research/wordsim353.html), which is one of the most popular test datasets used for this purpose. It contains word pairs together with human-assigned similarity judgments.
        >>> # 
        >>> # The word representations are evaluated by ranking the pairs according to their cosine similarities, and measuring the Spearmans rank correlation coefficient with the human judgments.
        >>> # 
        >>> # Let's look at the evaluation scores which are there in eval.json. For embeddings trained on the text8 dataset, scores above 0.65 are pretty good.

        >>> get_ipython().system('cat eval.json')


        >>> # Now, let us do a 2D visualization of the word vectors

        >>> import numpy as np
        >>> from sklearn.preprocessing import normalize

        >>> # Read the 400 most frequent word vectors. The vectors in the file are in descending order of frequency.
        >>> num_points = 400

        >>> first_line = True
        >>> index_to_word = []
        >>> with open("vectors.txt", "r") as f:
        >>>     for line_num, line in enumerate(f):
        >>>         if first_line:
        >>>             dim = int(line.strip().split()[1])
        >>>             word_vecs = np.zeros((num_points, dim), dtype=float)
        >>>             first_line = False
        >>>             continue
        >>>         line = line.strip()
        >>>         word = line.split()[0]
        >>>         vec = word_vecs[line_num - 1]
        >>>         for index, vec_val in enumerate(line.split()[1:]):
        >>>             vec[index] = float(vec_val)
        >>>         index_to_word.append(word)
        >>>         if line_num >= num_points:
        >>>             break
        >>> word_vecs = normalize(word_vecs, copy=False, return_norm=False)


        >>> from sklearn.manifold import TSNE

        >>> tsne = TSNE(perplexity=40, n_components=2, init="pca", n_iter=10000)
        >>> two_d_embeddings = tsne.fit_transform(word_vecs[:num_points])
        >>> labels = index_to_word[:num_points]


        >>> from matplotlib import pylab

        >>> get_ipython().run_line_magic('matplotlib', 'inline')


        >>> def plot(embeddings, labels):
        >>>     pylab.figure(figsize=(20, 20))
        >>>     for i, label in enumerate(labels):
        >>>         x, y = embeddings[i, :]
        >>>         pylab.scatter(x, y)
        >>>         pylab.annotate(
        >>>             label, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom"
        >>>         )
        >>>     pylab.show()


        >>> plot(two_d_embeddings, labels)


        >>> # Running the code above might generate a plot like the one below. t-SNE and Word2Vec are stochastic, so although when you run the code the plot won’t look exactly like this, you can still see clusters of similar words such as below where 'british', 'american', 'french', 'english' are near the bottom-left, and 'military', 'army' and 'forces' are all together near the bottom.

        >>> # ![tsne plot of embeddings](./tsne.png)

        >>> # ### Stop / Close the Endpoint (Optional)
        >>> # Finally, we should delete the endpoint before we close the notebook.

            # model, endpoint, and endpoint were created by predict.deploy so use the same name
        >>> sess.delete_endpoint(bt_endpoint.endpoint)
        >>> sess.delete_endpoint_config(bt_endpoint.endpoint_config)
        >>> sess.delete_model(bt_endpoint.endpoint)


    code: blazingtext_word2vec_subwords_text8.ipynb
       ->  'blazingtext_word2vec_text8.ipynb' with a few changes

       -> main difference, changes to hyperparameters: 'mode' to 'skipgram' (from 'batch_skipgram' and 'subwords' to 'True' 
       - with 'subwords=True', it will take longer to run/train

        >>> bt_model.set_hyperparameters(
        >>>     mode="skipgram",
        >>>     epochs=5,
        >>>     min_count=5,
        >>>     sampling_threshold=0.0001,
        >>>     learning_rate=0.05,
        >>>     window_size=5,
        >>>     vector_dim=100,
        >>>     negative_samples=5,
        >>>     batch_size=11,  #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
        >>>     evaluation=True,  # Perform similarity evaluation on WS-353 dataset at the end of training
        >>>     subwords=True,
        >>> )  # Subword embedding learning is not supported by batch_skipgram


       -> for inference words, use:

        >>> words = ["awesome", "awweeesome"]   # awesome with typos
            -> with subwords enabled, it will be still be able to generate a vector for 'awweeesome'
            -> return separately the vector for 'awesome' and 'awweeesome' so you can compare the vectors
        >>> words = ["awesome"]   # awesome with typos
        >>> words = ["awweeesome"]   # awesome with typos
            -> these vectors are similar, but NOT identical


    code: blazingtext_text_classification_dbpedia.ipynb

       -> 
       Introduction
       Text classification can be used to solve various use-casess like sentiment analysis, spam detection, hashtag prediction,
       etc. This notebook demonstrates the use of SageMaker BlazingText to perform supervised binary/multi class with single
       or multi label text classification. BlazingText can train the modelon more than a billion words in a couple of minutes
       using a multi-core CPU or a GPU, while acheiving performance on par with state-of-the-art deep learning text classification
       algorithms. BlazingText extends the fastText classifier to leverage GPU acceleration using custom CUDA kernels.


        # for each sentences, below are the classes to learn:
    >>> !cat dpedia_csv/classest.txt
         Company
         EducationInstitution
         Artist
         Athlete
         OfficeHolder
         MeanOfTransportation
         Building
         NaturalPlace
         Village
         Animal
         Plant
         Album
         File
         WrittenWork

         The following code creates the mapping from integer indices to class label wheih will later be used to retrieve
         the actual class name during inference

     >>> index_to_label = {}
     >>> with open("dbpedia_csv/classes.txt") as f:
     >>>     for i,label in enumerate(f.readlines()):
     >>>         index_to_label[str(i+1)] = label.strip()
     >>> print(index_to_label)
         { '1': 'Company', '2': 'EducationInstitution', '3': 'Artist', '4': 'Athlete', '5': 'OfficeHolder', 
           '6': 'MeanOfTransportation', '7': 'Building', '8': 'NaturalPlace', '9': 'Village', '10': 'Animal', 
           '11': 'Plant', '12': 'Album', '13': 'File', '14': 'WrittenWork'}



        . . .
        >>> bt_model = sagemaker.estimator.Estimator(
        >>>     container,
        >>>     role,
        >>>     instance_count=1,
        >>>     instance_type="ml.c4.2xlarge",
        >>>     train_volume_size=30,
        >>>     train_max_run=360000,
        >>>     input_mode="File",
        >>>     output_path=s3_output_location,
        >>>     sagemaker_session=sess,
        >>> )


        >>> # Please refer to [algorithm documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext_hyperparameters.html) for the complete list of hyperparameters.

        >>> bt_model.set_hyperparameters(
        >>>     mode="supervised",
        >>>     epochs=10,
        >>>     min_count=2,
        >>>     learning_rate=0.05,
        >>>     vector_dim=100,
        >>>     early_stopping=True,
        >>>     patience=4,
        >>>     min_epoch=5,  
        >>>     word_ngrams=2, 
        >>> )  

        >>> train_data = sagemaker.session.s3_input(s3_train_data, distribution="FullyReplicated",
        >>>     content_type="text/plain", s3_data_type="S3Prefix")
        >>> validation_data = sagemaker.session.s3_input(s3_validation_data, distribution="FullyReplicated",
        >>>     content_type="text/plain", s3_data_type="S3Prefix")
        >>> data_channels = {"train": train_data, 'validation': validation_data}

        >>>  bt_model.fit(inputs=data_channels, logs=True) 
 


        inferenciing
        >>> Sentences = ["Convair was an american aircraft manufacturing company which later expanded into rockets and spacecrafts."
        >>>                "Berwick secondary college is situated in the outer melbourne metropolitan suburb of berwick."]

        >>> tokenized_sentences = [' '.join(nltk.word_tokenize(sent)) for sent in sentences]

        >>> payload = {"instances" tokenized_sentences}

        >>> response = text_classifier.predict(json.dumps(payload))

        >>> predictions = json.loads(response)
        >>> print(json.dumps(prediction, indent=2))

        [
          { 
            { "prob": [ 0.988882040977478], 
              "label": [ __label__Compay" ]
            }
            { "prob": [ 0.997175633390731], 
              "label": [ __label__EducationalInstitution" ]
            }
           }
         ]
