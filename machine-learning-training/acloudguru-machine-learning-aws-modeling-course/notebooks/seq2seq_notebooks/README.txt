------------------------------------------------------
3.15 Examining SageMaker's Sequence-to-Sequence (Seq2Seq) Algorithm


  SageMaker sequence-to-sequence (Seq2Seq) algorithm Overview:
    - a supervised learning algorithm that uses a neural network architecture where a sequence of input
      tokens is transformed to another sequence of tokens as output.
    - Example applications include: machine translation (input a sentence from one language and predict what that
      sentence would be in another language), text summarization (input a longer string of words and predict a shorter
      string of words that is a summary), speech-to-text (audio clips converted into output sentences in tokens).

  Sequence-to-sequence algorithm's layers
     embedding layer
       - the encoded input tokens are mapped to a dense feature layer
       - It is a standard practice to initialize this embedding layer with pre-trained word vector,
         like FastText, and learn the parameters during the training process.
     encoder layer
       - compresses the input into a fixed-length feature vector.
       - Typically, an encoder is made of RNN network, like LSTM or GRU.
     decoder layer
       - converts the encoder feature to an output sequence of tokens.
       - This layer also is typically built with RNN architecture.


  Sagemaker Sequence-to-Sequence (Seq2Seq) algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq.html
    Learning Type:
      - supervised learning algorithm that specializes in language processing.
    File/Data Types:
      - training, test, and validation states expects data in RecordIO-Protobuf format
      - protobuf and JSON supported during the inference stage.
    Instance Type:
      - single GPU instances

    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext_hyperparameters.html
      No required hyperparameters

    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq-tuning.html
      validation:accuracy
         - Accuracy computed on the validation dataset.
         - goal: Maximize
      validation:bleu
        - BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has
          been machine-translated from one natural language to another.
        - BLEU score computed on the validation dataset.
        - Because BLEU computation is expensive, you can choose to compute BLEU on a random subsample of the validation
          dataset to speed up the overall training process.
        - Use the bleu_sample_size parameter to specify the subsample.
        - goal: Maximize
      validation:perplexity
        - perplexity is a loss function computed on the validation dataset.
        - Perplexity measures the cross-entropy between an empirical sample and the distribution predicted by a model
          and so provides a measure of how well a model predicts the sample values,
        - Models that are good at predicting a sample have a low perplexity.
        - goal: Minimize

  Sequence-to-Sequence (Seq2Seq) Business use cases
    language translations
      - to translate it in sequence of words from one language to another language.
    speech-to-text conversion.
      - given an audio vocabulary, you can then predict the textual representation of those spoken words.
    code generation and auto completion
      - assist developers by generating code snippets or completing the code based on content.

   Note: Missing:
     Finally, in the Resources section, you will see a page containing a sample notebook demonstrating SageMaker's
     sequence-to-sequence algorithm.


     Sequence-to-Sequence Sample Notebooks
       - For a sample notebook that shows how to use the SageMaker AI Sequence to Sequence algorithm to train
         a English-German translation model, see Machine Translation English-German Example Using SageMaker AI Seq2Seq


         Machine Translation English-German Example Using SageMaker Seq2Seq
         https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/seq2seq_translation_en-de/SageMaker-Seq2Seq-Translation-English-German.html
            Save to: notebooks/seq2seq_notebooks/SageMaker-Seq2Seq-Translation-English-German.ipynb.txt

