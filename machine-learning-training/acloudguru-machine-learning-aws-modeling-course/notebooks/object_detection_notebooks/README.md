------------------------------------------------------
3.17 Reviewing SageMaker's Object Detection Algorithm

  SageMaker's Object Detection algorithm Overview
    - It is a supervised learning algorithm that takes images as input, and identifies all instances of objects
      within the image scene.

    - The object is categorized into one of the classes in a specified collection with a confidence score that
      it belongs to the class.
    - Its location and scale in the image are indicated by a rectangular bounding box

    - It uses the Single Shot multibox Detector (SSD) framework and supports two base networks: VGG and ResNet.

    - Supoorts full training mode and transfer learning mode


  Sagemaker Object Detection algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html
    Learning Type:
      - supervised learning algorithm that specializes in classifying and detecting images.

    File/Data Types:
      - recordIO format or image formats (JPG or PNG).
      - Each image also needs a .json file for annotation, and the file name must match the image name.
        - annotations provides object classes and locations for each image

    Instance Type:
      - recommends using GPU instances with more memory for training the images
      - CPU or a GPU can be used in the inference stage.

    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection-api-config.html
      required hyperparameters
        num_classes
          - Number of output classes.
          - This parameter defines the dimensions of the network output and is typically set to the number of classes in the dataset.
          - Valid values: positive integer
        num_training_samples
          - Number of training examples in the input dataset.
          - If there is a mismatch between this value and the number of samples in the training set, then the behavior
            of the lr_scheduler_step parameter is undefined and distributed training accuracy might be affected.
          - Valid values: positive integer

    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection-tuning.html
        validation:mAP
          - Mean Average Precision (mAP) computed on the validation set.
          - goal: Maximize

  Object Detection Business use cases

    automated checkout system
       - to detect and identify items in a shopping cart at a checkout counter.
    manufacturing quality control
       - to detect defects in the products during the manufacturing process
    finance
      - to detect and verify information in scanned documents like checks, invoices and ID cards.

  Note: Missing:
    Finally, in the resources section, you will see a page containing a sample notebook demonstrating SageMaker's
    object detection algorithm.

    Object Detection Sample Notebooks
      - For a sample notebook that shows how to use the SageMaker AI Object Detection algorithm to train and host
        a model on the Caltech Birds (CUB 200 2011) dataset using the Single Shot multibox Detector algorithm,
        see Amazon SageMaker AI Object Detection for Bird Species

        Amazon SageMaker Object Detection for Bird Species
          https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/object_detection_birds/object_detection_birds.html
          Saved to: notebooks/object_detection_notebooks/object_detection_birds.ipynb


