------------------------------------------------------
7.16 Creating an MXNet Image Classifier in Amazon SageMaker


  Resources

    Note: Downloaded demo files to:
      C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\7_16_creating_an_mxnet_image_classifier_demo

     -> Jupyter Notebook:
        CreateAnMXNetImageClassifier.ipynb
        -> original notebook: need to rename without '.orig' to use else you may have a load error
           CreateAnMXNetImageClassifier.orig.ipynb
     -> dataset data (pickled files):
          lego-simple-mx-train
          lego-simple-mx-test


  python pickle:
     - The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
     - "Pickling" is the process whereby a Python object hierarchy is converted into a byte stream, and
       "unpickling" is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted
        back into an object hierarchy.

  mxnet transforms
    Gluon provides pre-defined vision transformation and data augmentation functions in the mxnet.gluon.data.vision.transforms module.
    https://mxnet.apache.org/versions/1.5.0/tutorials/gluon/transforms.html
  mxnet transforms.compose()
    With Compose we can choose and order the transforms we want to apply.
  mxnet transforms.compose ToTensor & Normalize:
    transforms.Compose([ transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])
    Normalize
     - We scaled the values of our data samples between 0 and 1 as part of ToTensor but we may want or need to normalize
       our data instead: i.e. shift to zero-center and scale to unit variance


Frameworks

For this lab, we will be using Apache MXNet to build and train a model to classify images, and specifically take advantage of the Gluon API provided with MXNet to make that process really easy.
MXNet

A flexible and efficient library for deep learning.

    MXNet provides optimized numerical computation for GPUs and distributed ecosystems, from the comfort of high-level environments like Python and R.
    MXNet automates common workflows, so standard neural networks can be expressed concisely in just a few lines of code.

(Source: https://mxnet.apache.org/)
Gluon

Based on the the Gluon API specification, the Gluon library in Apache MXNet provides a clear, concise, and simple API for deep learning. It provides basic building blocks for neural networks that make it easy to prototype, build, and train deep learning models without sacrificing training speed. Install a recent version of MXNet to get access to Gluon.

(Source: AWS Blog)



Apache MXNet is an open-source machine learning framework focusing on deep learning with neural networks.
In this lab, you will use MXNet to create a neural network that performs a basic image classification task:
deciding which LEGO brick is in an image to help you sort your giant pile of blocks. MXNet supports many
programming languages, but we will use Python.

Learning objectives
  - Navigate to the Jupyter Notebook
  - Load and Prepare the Data
  - Train the MXNet Model
  - Evaluate the Model
  - Make a Batch Prediction


  Creating an MXNet Image Classification in SageMaker Flow:

                             Lego
  SageMaker Notebook ------- Image -------------------------->  MXNet
                                                                Apache MXNet

                                                                  Model
     SageMaker                          Datasets
                          Training 80%           Testing 20%

      ------------------------------------------------------
Solution
Log in to the Lab Environment

    To avoid issues with the lab, open a new Incognito or Private browser window to log in to the lab. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region.

Navigate to the Jupyter Notebook

    In the search bar on top, type "Sagemaker" to search for the Sagemaker service.
    Click on the Amazon SageMaker result to go directly to the Sagemaker service.
    Click on the Notebook Instances button to look at the notebook provided by the lab.
    Check to see if the notebook is marked as InService. If so, click on the Open Jupyter link under Actions.
    Click on the CreateAnMXNetImageClassifier.ipnyb file.
    When prompted, select the conda_python3 kernel. In the notebook, the code line of !pip install mxnet installs the needed mxnet package.
    Wait for the kernel to spin up. You'll see a green button that reads Kernel ready in the upper right momentarily when the kernel is finished spinning up.

Load and Prepare the Data

    Under 1) Import Libraries, select the cell containing the installation command for Apache MXNet, and run the cell either by clicking the Run button in the menu or press Ctrl + Enter to run the cell.

    Run the second cell to import the necessaries libraries for this lab, including the now-installed Apache MXNet

    Run the third cell to set the random seed for the lab to ensure reproducibility.

    Run the fourth cell to tell MXNet that it's using a CPU rather than a GPU for this lab.

    Under Load the Data, update and run the first cell to load the training and testing images and labels into an NDArray using Pickle. The images and labels are provided in the lego-simple-mx-train file and the lego-simple-mx-test file.

    train_fh = open('lego-simple-mx-train', 'rb')
    test_fh = open('lego-simple-mx-test', 'rb')

    train_data = pickle.load(train_fh)
    test_data = pickle.load(test_fh)

    Run the second cell to Add in the human-readable class names for the labels.

    Under Convert to MXNet Tensors, run the first cell to convert the training and testing NDArrays to MXNet Tensors. For better results, normalize the data using the mean of 0.13 and standard deviation of 0.31, which have been precomputed for this dataset.

    Run the second cell to visualize the first few images from the training data set to better understand the data.

    Run the third cell to look at more of the data formatted into a graph.

Train the MXNet Model

    Under 3) Create the Model, update and run the first cell to create a neural network model with one flatten layer and three dense layers using Gluon.

    net = nn.HybridSequential(prefix='MLP_')
    with net.name_scope():
        net.add(
            nn.Flatten(),
            nn.Dense(128, activation='relu'),
            nn.Dense(64, activation='relu'),
            nn.Dense(10, activation=None)
        )

    Run the second cell to load the data using Gluon's data loader with a batch size of 34.

    Run the third cell to initialize the model.

    Run the fourth cell to create a trainer object and use it to maintain the state of the training.

    Run the fifth cell to define the accuracy metric, keep track of accuracy during the training process, and choose a loss function appropriate for classification tasks.

    Run the sixth cell to train the model using the training data and training labels for 10 epochs, and save the history of the training process.

Evaluate the Model

    Under 4) Evaluate the Model, run the first cell to calculate the accuracy of the model on the testing data and output the data as a graph.

    Update and run the second cell to apply the model to the testing data.

    test_loader = mx.gluon.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

    Run the third cell to measure the accuracy of the model.

Test the Model

    Under 5) Test the Model, run the cell to define a couple of functions to display the model's results as a graph.

    Under Single Prediction, run the first cell to choose an image from the test set.

    Update and run the second cell to make a prediction using the model.

    predictions_single = net(prediction_image)
    predictions_single

    Run the third cell to look at the resulting predictions as a bar chart.

    Run the fourth cell to visualize the highest-accuracy prediction as an image.

Make a Batch Prediction

    Under Batch Prediction, run the first cell to predict the classes for 16 images from the testing data.
    Run the second cell to compare the predictions against the actual data.

Conclusion

Congratulations - you've completed this hands-on lab!

      ------------------------------------------------------

