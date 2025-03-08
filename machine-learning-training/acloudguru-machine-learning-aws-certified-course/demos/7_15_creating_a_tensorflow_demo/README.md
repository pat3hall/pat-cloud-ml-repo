------------------------------------------------------
7.15 Creating a TensorFlow Image Classifier in Amazon SageMaker

  Resources

    Note: Downloaded demo files to:
      C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\7_15_creating_a_tensorflow_demo

     -> Jupyter Notebook:
        CreateATensorFlowImageClassifier_completed.ipynb
     -> dataset data:
          lego-simple-train-images.npy
          lego-simple-train-labels.npy
          lego-simple-test-images.npy
          lego-simple-test-labels.npy

TensorFlow

  - TensorFlow is an end-to-end, open source platform for machine learning.
  - It has a comprehensive, flexible ecosystem of tools, libraries, and community resources.
  - This lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.
    (Source: https://www.tensorflow.org/)

Keras

  - Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
  - It was developed with a focus on enabling fast experimentation.
  - Being able to go from idea to result with the least possible delay is key to doing good research.
    (Source: https://keras.io/)

Or, to put it another way:
  - TensorFlow is a complex and powerful tool, but Keras provides a high-level interface that allows us to rapidly
    prototype models without dealing with all of the complexity TensorFlow offers.

About this lab

TensorFlow is the biggest name in machine learning frameworks. In this lab, you will use TensorFlow to create
a neural network that performs a basic image classification task: deciding which LEGO brick is in an image
to help you sort your giant pile of blocks.

Learning objectives
  - Navigate to the Jupyter Notebook
  - Load and Prepare the Data
  - Train the TensorFlow Model
  - Evaluate the Model
  - Make a Batch Prediction


  Creating a TensorFlow Image Classification in SageMaker Flow:

                             Lego
  SageMaker Notebook ------- Image --------------------------> Keras   --- 2x3 Brick (94.2%) ---> TensorFlow


     SageMaker                          Datasets
                          Training 80%           Testing 20%


     Notes:
        TensorFlow: underlying framework
        Keras:      API

     ------------------------------------------------------
Creating a TensorFlow Image Classifier in AWS SageMaker
Introduction

TensorFlow is the biggest name in machine learning frameworks. In this lab, you will use TensorFlow to create a neural network that performs a basic image classification task: deciding which LEGO brick is in an image to help you sort your giant pile of blocks.
Solution
Log in to AWS Console

    Open a new Incognito or Private browser window to log in to the lab. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region.

Navigate to the Jupyter Notebook

    In the search bar, type "Sagemaker" to search for the Sagemaker service.
    Click on the Amazon SageMaker result to go directly to the Sagemaker service.
    Click on the Notebook Instances button to look at the notebook provided by the lab.
    Check to see if the notebook is marked as InService. If so, click on the Open Jupyter link under Actions.
    Click on the CreateATensorFlowImageClassifier.ipnyb file.
    Wait for the kernel to spin up. You'll see a green button that reads Kernel ready in the upper right momentarily when the kernel is finished spinning up.

Load and Prepare the Data

    Make sure your kernel can support TensorFlow 2.
        Check the bar in the upper right corner to see if it contains tensorflow2.
        If it does not, click on Kernel in the menu, select Change kernel, and then select conda_tensorflow2_p36 from the list. The version number available to you may differ.

    Under 1) Import Libraries, select the cell containing the import code and click the Run button in the menu or press Ctrl + Enter to run the cell.
        You will see a warning that no CUDA-capable devices are detected. This is expected, since our notebook doesn't have access to any GPU's, and won't impact our lab.

    Under 2) Load the Data, update the code as below, and run the first code cell to load the training images and labels into numpy arrays. The images and labels are provided in the respective files.

    train_images = np.load('lego-simple-train-images.npy')
    train_labels = np.load('lego-simple-train-labels.npy')
    test_images = np.load('lego-simple-test-images.npy')
    test_labels = np.load('lego-simple-test-labels.npy')

    Run the second code cell to add in the human-readable class names for the labels.

    Run the rest of the code cells to visualize the first few images from the training data set and testing set to better understand the data.

Train the TensorFlow Model

    Under Create the Model, update and run the first code cell to create a flatten layer and two dense layers for a neural network model using Keras.

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(48,48)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    Update and run the second code cell to compile the model, using adam as the optimizer, sparse categorical cross-entropy for the loss function, and accuracy as the metric.

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    Update and run the third code cell to train the model using the training data and training labels.

    history = model.fit(train_images, train_labels, epochs=4)

    Run the fourth code cell to save and plot the history of the training process in terms of accuracy values and loss values.

Test and Analyze the Model

    Under Evaluate the Model, update and run the first code cell to calculate the loss and accuracy of the model on the testing data.

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    Under Single Prediction, run the first code cell to pick a random image in the test set.

    Run the next code cell to transform the image into a collection of one image.

    Run the third code cell to pass the image into the predict method.

    predictions_single = model.predict(img)
    predictions_single

    Run the fourth code cell to use the argmax function to find the highest-probability prediction result found by the predict method.

    Run the fifth code cell to look up the class name of the prediction result and obtain the model's probability of being correct.

    Run the sixth code cell to determine the label that the model predicted for the image and compare that to the actual label.

    Run the seventh code cell to run functions to display the image and the results as a graph.

    Run the eighth code cell to run a function to display the results as a bar chart.

Make a Batch Prediction Using the Testing Data

    Under Batch Prediction, run the first and second code cell to predict the labels for all of the test images.
    Run the third code cell to summarize the results with bar charts measuring the probability of the predictions for the first 16 images of our test data.

Conclusion

Congratulations - you've completed this hands-on lab!
     ------------------------------------------------------


