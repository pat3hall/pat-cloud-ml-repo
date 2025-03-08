------------------------------------------------------
3.18 Exploring SageMaker's Semantic Segmentation

  SageMaker's Semantic Segmentation Algorithm
    - It is a supervised learning algorithm that tags every pixel in an image with a class label from a
      predetermined set of classes.

    - Since the tagging is done at the pixel level, this algorithm provides information about the shapes
      of the objects present in the image.
    - The output produced by this algorithm is represented as a grayscale image, also called a segmentation mask.

  Image Processing Algorithms
    Image classification
      - analyzes all the images and classifies them into one or more multiple output categories.
    Object detection algorithm
      - detects and classifies all instances of an object present in an image
      - It indicates the location and scale of each object in the image with a rectangular bounding box
    Semantic Segmentation
      - classifies every pixel present in an image.
      - it also provides information about the shapes of the objects contained in the image


  Sagemaker Semantic Segmentation algorithm Attributes
    https://docs.aws.amazon.com/sagemaker/latest/dg/semantic-segmentation.html
    Learning Type:
      - supervised learning algorithm that specializes in as assigning class labels to images at pixel level

    File/Data Types:
      - the data to be provided in four separate channels, two for images and two for annotations:
          train (jpg), train_annotation (png), validation (jpg), and validation_annotation (png)
      - Annotations are expected to be uncompressed PNG images
      - Every JPG image in the train and validation directories have a corresponding PNG label image with
        the same name in the train_annotation and validation_annotation directories.

    Instance Type:
      - recommends using GPU instances with more memory for training the images
      - CPU or a GPU can be used in the inference stage.

    Hyperparameters
      https://docs.aws.amazon.com/sagemaker/latest/dg/segmentation-hyperparameters.html
      required hyperparameters
        num_classes
          - Number of output classes to segment
          - Valid values:  2 ≤ positive integer ≤ 254
        num_training_samples
          - The number of samples in the training data.
          - The algorithm uses this value to set up the learning rate scheduler.
          - Valid values: positive integer

    Metrics
      https://docs.aws.amazon.com/sagemaker/latest/dg/semantic-segmentation-tuning.html
        validation:mIOU
          - IoU = (Area of overlap) / (Area of Union);  and MIoU is the mean IoU
          - The area of the intersection of the predicted segmentation and the ground truth divided by the
            area of union between them for images in the validation set. Also known as the Jaccard Index.
          - goal: Maximize

        validation:pixel_accuracy
          - The percentage of pixels that are correctly classified in images from the validation set.
          - goal: Maximize

  Semantic Segmentation Business use cases

    satellite and aerial imagery
      - to segment different land cover types, such as urban areas or water bodies.
    retail
      - to segment products on retail shelves.
    entertainment and media
      - to segment inappropriate or harmful content in images and videos.


  Semantic Segmentation Sample Notebooks
    https://docs.aws.amazon.com/sagemaker/latest/dg/semantic-segmentation.html#semantic-segmentation-sample-notebooks
    - For a sample Jupyter notebook that uses the SageMaker AI semantic segmentation algorithm to train a model
      and deploy it to perform inferences, see the Semantic Segmentation Example.
      https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/semantic_segmentation_pascalvoc/semantic_segmentation_pascalvoc.html

      Amazon SageMaker Semantic Segmentation Algorithm

      Saved to: notebooks/semantic_segmentation_notebooks/semantic_segmentation_pascalvoc.ipynb.txt


