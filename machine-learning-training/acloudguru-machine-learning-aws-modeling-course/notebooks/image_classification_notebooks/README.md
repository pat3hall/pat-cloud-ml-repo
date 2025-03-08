------------------------------------------------------
3.16 Discovering SageMaker's Image Classification Algorithm

  SageMaker's image classification algorithm Overview:
    - It is a supervised learning algorithm that accepts labeled images as input and classifies them into one of
      the output categories.
    - uses a CNN to train from scratch or leverage transfer learning when the training data is limited.

  Image classification modes
    full training mode,
      - the image training is performed from the scratch by initializing the network with random weights and
        training requires a large data set.
    transfer learning mode,
      - can leverage previously trained images and the network is initialized with pre-trained weights.
      - the training can be achieved with a smaller dataset.

   CNN Architecture
     - typically will contain many layers.

                                                                                         Fully-
      input   ------>  Convolutional  ------> activation  ------> Pooling Layer ------> Flattening  ----> Connected
      layer                layers               fcn                 Layer(s)              Layer           Layer


    convolution layer
      - to extract features using learn filters.
    activation function
      - applies a nonlinear activation function like ReLU
    pooling layer
      - reduces spatial dimensions, which prevents overfitting.
    flattened layer
      - converts the 2D matrix to a 1D vector.
    fully connected layer
      - performs a classification task.


  Sagemaker Image Classification algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html
    Learning Type:
      - supervised learning algorithm that specializes in classifying images
    File/Data Types:
      - recordIO format or image formats (JPG or PNG)
      - requires a single tab separated .LST file that contains a list of image files.
    Instance Type:
      - recommends using GPU instances with more memory for training the images
      - CPU or a GPU can be used in the inference stage.

    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/IC-Hyperparameter.html
      required hyperparameters
        num_classes
          - Number of output classes.
          - This parameter defines the dimensions of the network output and is typically set to the number of classes in the dataset.
          - Besides multi-class classification, multi-label classification is supported too.
          - Please refer to Input/Output Interface for the Image Classification Algorithm for details on how to work with
            multi-label classification with augmented manifest files.
          - Valid values: positive integer
        num_training_samples
          - Number of training examples in the input dataset.
          - If there is a mismatch between this value and the number of samples in the training set, then the behavior
            of the lr_scheduler_step parameter is undefined and distributed training accuracy might be affected.
          - Valid values: positive integer

    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/IC-tuning.html
      validation:accuracy
         - The ratio of the number of correct predictions to the total number of predictions made.
         - goal: Maximize

  Image Classification Business use cases
     healthcare
       - to classify medical images like X-rays and CT scans to assist in diagnosing diseases.
     autonomous vehicles
       - to classify the moving objects like pedestrians, vehicles, animals in the images captured by the vehicle cameras.
     security
       - to classify images from surveillance cameras, to detect unauthorized persons or suspicious activities.


    Image Classification Sample Notebooks
      - For a sample notebook that uses the SageMaker AI image classification algorithm, see Build and Register an
        MXNet Image Classification Model via SageMaker Pipelines
          https://github.com/aws-samples/amazon-sagemaker-pipelines-mxnet-image-classification/blob/main/image-classification-sagemaker-pipelines.ipynb
            Saved to: notebooks/image_classification_notebooks/image-classification-sagemaker-pipelines

