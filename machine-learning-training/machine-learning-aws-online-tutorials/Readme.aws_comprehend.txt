--------------------------------------------
  AWS Comprehend
--------------------------------------------

Introduction to Amazon Comprehend in 5 minutes! - Dabble Lab 272
https://www.youtube.com/watch?v=NR-PeUad2XI
--------------------------------------------

Classify documents with Amazon Comprehend - Part 1 | Amazon Web Services
https://www.youtube.com/watch?v=QsqHvDPRSSQ

Summary:

Use Amazon Comprehend custom classification to organize your documents into categories (classes) 
that you define. Amazon Comprehend is an AI service from AWS, which uses natural language 
processing (NLP) to derive and understand valuable insights from text within documents.

  Comprehend
    - a fully managed, continuously trained, AWS AI service that can derive and understand valuable
      insights from texts within documents

  Intelligent document processing solutions
    - extract data to support automation of high volume repetitive document processing tasks for
      analysis and insight
    - use natural language technologies and computer vision to extract data from structured and
      unstructured content especially from documents to support automation and augmentation
   Using Comprehend for Intelligent document processing
    - Comprehend helps you with document processing with no machine learning experience
    - you can use classificaiton and extraction capabilities to rapidly process a variety of
      document types and accurately extract insights
    - Comprehend can detect and protect sensitive data and help meet compliance requirements

   Intelligent Document Processing (IDP) workflow for the insurance industry

     documents --> Secure  --> Document       --> Document   --> Validation &  --> downstream
                   storage     Classification     Extraction     Business rules    systems
                     ^            ^               ^     ^    --> Human review 
      ---------------|------------|---------------|-----|--------------------
      |              |            |               |     |                   |
      |          Classification   NER ------------|   Pre-trained           |
      |           model           model               PII                   |
      |            ^               ^                  model                 |
      |            |               |                                        |
      |          Train /          Train /                                   |
      |          Re-train         Re-train          training                |
      |          custom           custom NER        data                    |
      |          classification   model                                     |
      |          model                                                      |
      -----------------------------------------------------------------------


  Custom Classification
    - two step processs:
      1. train a custom classification model called a classifier to recognize the classes that
         are of interest to you
         - use Comprehend using an existing set of label documents in one of two ways:
            a. CSV: contains the label for each document and the corresponding text from these docs
            b. Augmented manifest file: from SageMaker Ground Truth data labeling service
         - output is a trained model to use on unlabel docs
      2. Use your model to classify any number of unlabeled document sets


   common document types:
     - PDF, TIFF, PNG, JPG
     - examples: claim forms, driver's license, insurance ID, policy documents, claim quotes

    Previous Custom Classification flow
     step 1:
     - Extract text from documents using Optical Character recognition (OCR) or text parsing technology
     step 2:
       - use Comprehend trained classifier model to classify the docs, either in real-time with 
         real-time endpoint or with batch processing

    New Comprehend Custom Classification flow
      - Comprehend includes Amazon Textract and digital PDF parser to convert native docs to text
      - now, native PDF, TIFF, PNG, JPG documents can be passed to Comprehend
      - just specify to Comprehend "Document read mode"


   Comprehend Custom Classification flow

   AWS -> comprehend -> Custom Classification <left tab>


     Custom classification
       - Build and train models to classify your documents with custom categories or labels.
     Real-time flow:
       Purchase endpoint
         - Purchase one or more endpoints for your model to enable synchronous analysis requests.
       Custom real-time analysis
         Select an endpoint to use your model to analyze your document in real time.
     batch flow:
       Batch analysis
         - Create asynchronous custom classification jobs to classify documents using custom 
           categories or labels.

  Create Custom Model options:
    Training model type
      Plain text documents
        - Choose this option if you labeled plain-text documents for CSV or augmented manifest.
     Native documents (PDF, Word, images)
       - Choose this option if you labeled PDF, Word documents or images for CSV.
    Data format
      - To train your custom model, you must provide training data. 
      - This data must be formatted as either a CSV file or as one or more augmented manifest files.
      CSV file Info
        - The CSV file that contains either the annotations or the entity lists for your training data. 
        - The required format depends on the type of CSV file that you provide.
      Augmented manifest Info
       - A labeled training dataset that is produced by Amazon SageMaker Ground Truth. 
       - You can provide up to 5 augmented manifest files. To create an augmented manifest file, you can 
         create a labeling job in Amazon SageMaker Ground Truth.
    Classifier mode
     Using Single-label mode
       - The training data file must have one class and one document on each line. 
       - It must have at least 10 documents for each class.
     Using Multi-label mode
       - The training data file must have one or more classes and one document on each line. 
       - It must have at least 10 documents for each class.
    Training dataset
      - A training dataset teaches your model to classify. 
      - Paste the URL of an input data file in S3, or select a bucket or folder location in S3.


   Real-time endpoints use:
     - best for low latency use cases with single page documents
   Batch use
     - when processing large number of multi-page documents, recommend using asynchronous analysis
       [batch] for classification

   Comceptual IDP workflow:

   -> Documents  --> S3    --> Lambda    --> Comprehend   --> S3      -->  Lambda       ---
      PDF, TIFF,     docs      Classify      custom           output       process        |
      PNG, JPG       bucket    documents     Classifier       bucket       Classification |
                                                                         results          |
                                                                                          |
                                                                                          |
                                             Comprehend  <--   Lambda   <--   S3     <-----
                                             Extraction        Extract &      Classified docs
                                              (NER)*           enrich docs    sorted by prefix

  * low confidence data sent to human reviewres

  * NER: Named Entity Recognition
       - 
--------------------------------------------
AWS Comprehend FAQs
  https://aws.amazon.com/comprehend/faqs/

What is Amazon Comprehend?
  - Amazon Comprehend is a natural language processing (NLP) service that uses machine learning to find meaning 
    and insights in text.

What can I do with Amazon Comprehend?
  - You can use Amazon Comprehend to identify the language of the text, extract key phrases, places, people, 
    brands, or events, understand sentiment about products or services, and identify the main topics from 
    a library of documents. 
  - The source of this text could be web pages, social media feeds, emails, or articles. 
  - You can also feed Amazon Comprehend a set of text documents, and it will identify topics (or group of 
    words) that best represent the information in the collection. 
  - The output from Amazon Comprehend can be used to understand customer feedback, provide a better search 
    experience through search filters and uses topics to categorize documents.


The most common use cases include:

  Voice of customer analytics: 
    - You can gauge whether customer sentiment is positive, neutral, negative, or mixed based on the feedback 
      you receive via support calls, emails, social media, and other online channels.

  Semantic search: 
    - You can use Amazon Comprehend to provide a better search experience by enabling your search engine to 
      index key phrases, entities, and sentiment. 
    - This enables you to focus the search on the intent and the context of the articles instead of basic 
      keywords.

  Knowledge management and discovery: 
    - You can analyze a collection of documents and automatically organize them by topic. 
    - You can then use the topics to personalize content for your customers.

  Cost:
    https://aws.amazon.com/comprehend/pricing/
    Natural language processing: 
      - Amazon Comprehend APIs for entity recognition, sentiment analysis, syntax analysis, key phrase 
        extraction, and language detection can be used to extract insights from natural language text. 
      - These requests are measured in units of 100 characters (1 unit = 100 characters), with a 3 unit 
        (300 character) minimum charge per request.

--------------------------------------------
Comprehend Boto3 documentation
  https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehend.html

  # create comprehend client
  import boto3
  client = boto3.client('comprehend')

  [Some of ] These are the available methods:
    batch_detect_dominant_language
    batch_detect_entities
    batch_detect_key_phrases
      Request Syntax

      response = client.batch_detect_key_phrases(
          TextList=[
              'string',
          ],
          LanguageCode='en'|'es'|'fr'|'de'|'it'|'pt'|'ar'|'hi'|'ja'|'ko'|'zh'|'zh-TW'
      )

      Response Syntax
      
      {
          'ResultList': [
              {
                  'Index': 123,
                  'KeyPhrases': [
                      {
                          'Score': ...,
                          'Text': 'string',
                          'BeginOffset': 123,
                          'EndOffset': 123
                      },
                  ]
              },
          ],
          'ErrorList': [
              {
                  'Index': 123,
                  'ErrorCode': 'string',
                  'ErrorMessage': 'string'
              },
          ]
      }
      
    batch_detect_sentiment
    batch_detect_syntax
    batch_detect_targeted_sentiment
    classify_document
    close
    contains_pii_entities

--------------------------------------------




