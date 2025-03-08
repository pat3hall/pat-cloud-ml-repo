-------------------------------------------------------------------------------------
Ditectrev /
Amazon-Web-Services-Certified-AWS-Certified-Machine-Learning-MLS-C01-Practice-Tests-Exams-Question 
  https://github.com/Ditectrev/Amazon-Web-Services-Certified-AWS-Certified-Machine-Learning-MLS-C01-Practice-Tests-Exams-Question?tab=readme-ov-file#table-of-contents
-------------------------------------------------------------------------------------


Question 2:
A Machine Learning Specialist is designing a system for improving sales for a company. The objective is to use the large 
amount of information the company has on users' behavior and product preferences to predict which products users would 
like based on the users' similarity to other users. What should the Specialist do to meet this objective?

  choices:
    Build a content-based filtering recommendation engine with Apache Spark ML on Amazon EMR.
  x Build a collaborative filtering recommendation engine with Apache Spark ML on Amazon EMR.
    Build a model-based filtering recommendation engine with Apache Spark ML on Amazon EMR.
    Build a combinative filtering recommendation engine with Apache Spark ML on Amazon EMR.



-------------------
Approaches to build Recommender Systems
https://analyticsindiamag.com/ai-mysteries/collaborative-filtering-vs-content-based-filtering-for-recommender-systems/

Collaborative Filtering
  - based on the past interactions that have been recorded between users and items, in order to produce new recommendations
  - Collaborative Filtering tends to find what similar users would like and the recommendations to be provided and in order 

Model based filtering and combinative filtering do not appear in my searches
    to classify the users into clusters of similar types and recommend each user according to the preference of its cluster. 
 - The standard method used by Collaborative Filtering is known as the Nearest Neighborhood algorithm


Content Based filtering
  - The content-based approach uses additional information about users and/or items. 
  - This filtering method uses item features to recommend other items similar to what the user likes and also based on 
  - If we consider the example for a movies recommender system, the additional information can be, the age, the sex, the 
  - The Content-based approach requires a good amount of information about items’ features, rather than using the user’s 
    interactions and feedback.
    job or any other personal information for users as well as the category, the main actors, the duration or other characteristics 
    for the movies i.e the items.
    their previous actions or explicit feedback



-------------------------------------------------------------------------------------
Question 3:
A Mobile Network Operator is building an analytics platform to analyze and optimize a company's operations using Amazon Athena and 
Amazon S3. The source systems send data in .CSV format in real time. The Data Engineering team wants to transform the data to the 
Apache Parquet format before storing it on Amazon S3. Which solution takes the LEAST effort to implement?

    Ingest .CSV data using Apache Kafka Streams on Amazon EC2 instances and use Kafka Connect S3 to serialize data as Parquet
    Ingest .CSV data from Amazon Kinesis Data Streams and use Amazon Glue to convert data into Parquet.
    Ingest .CSV data using Apache Spark Structured Streaming in an Amazon EMR cluster and use Apache Spark to convert data into Parquet.
  x Ingest .CSV data from Amazon Kinesis Data Streams and use Amazon Kinesis Data Firehose to convert data into Parquet.


Note: Although Glue can directly convert CSV to Parquet, it would not be a near real-time solution 

-------------------
Convert input data format in Amazon Data Firehose
  https://docs.aws.amazon.com/firehose/latest/dev/record-format-conversion.html

  - Amazon Data Firehose can convert the format of your input data from JSON to Apache Parquet or Apache ORC before storing the data in Amazon S3. 
  - Parquet and ORC are columnar data formats that save space and enable faster queries compared to row-oriented formats like JSON. 
  - If you want to convert an input format other than JSON, such as comma-separated values (CSV) or structured text, you can use AWS Lambda 
    to transform it to JSON first. For more information, see Transform source data in Amazon Data Firehose.
    https://docs.aws.amazon.com/firehose/latest/dev/data-transformation.html

-------------------
AWS Certified Machine Learning - Specialty (MLS-C01)
2.9 Quiz AWS Certified Machine Learning - Specialist 2020 - Data Collection Quiz

Question 1 (retry)

You have been tasked with converting multiple JSON files within a S3 bucket to Apache Parquet format. Which AWS service can 
you use to achieve this with the LEAST amount of effort?


Correct Answer

   - AWS Glue makes it super simple to transform data from one format to another. You can simply create a job that takes 
     in data defined within the Data Catalog and outputs in any of the following formats: avro, csv, ion, grokLog, json, 
     orc, parquet, glueparquet, or xml.


-------------------------------------------------------------------------------------
Question 6:
A Machine Learning Specialist is using an Amazon SageMaker notebook instance in a private subnet of a corporate VPC. The ML 
Specialist has important data stored on the Amazon SageMaker notebook instance's Amazon EBS volume, and needs to take a snapshot 
of that EBS volume. However, the ML Specialist cannot find the Amazon SageMaker notebook instance's EBS volume or Amazon EC2 
instance within the VPC. Why is the ML Specialist not seeing the instance visible in the VPC?

    Amazon SageMaker notebook instances are based on the EC2 instances within the customer account, but they run outside of VPCs.
    Amazon SageMaker notebook instances are based on the Amazon ECS service within customer accounts.
  x Amazon SageMaker notebook instances are based on EC2 instances running within AWS service accounts.
    Amazon SageMaker notebook instances are based on AWS ECS instances running within AWS service accounts.

-------------------

  Amazon SageMaker notebooks
   - Fully managed notebooks in JupyterLab for exploring data and building ML models
   - Notebooks are a fully managed service, so you do not have access to notebook EBS volume images

-------------------------------------------------------------------------------------
Question 10:
A Machine Learning Specialist has completed a proof of concept for a company using a small data sample, and now the Specialist is ready 
to implement an end-to-end solution in AWS using Amazon SageMaker. The historical training data is stored in Amazon RDS. Which approach 
should the Specialist use for training a model using that data?

    Write a direct connection to the SQL database within the notebook and pull data in.
  x Push the data from Microsoft SQL Server to Amazon S3 using an AWS Data Pipeline and provide the S3 location within the notebook.
    Move the data to Amazon DynamoDB and set up a connection to DynamoDB within the notebook to pull data in.
    Move the data to Amazon ElastiCache using AWS DMS and set up a connection within the notebook to pull data in for fast access.

-------------------

 Migrate workloads from AWS Data Pipeline
   https://aws.amazon.com/blogs/big-data/migrate-workloads-from-aws-data-pipeline/
   Note: After careful consideration, we have made the decision to close new customer access to AWS Data Pipeline, effective July 25, 2024.

   -> for new customers, the path would be AWS Glue to S3 instead AWS Data Pipeline


-------------------------------------------------------------------------------------
Question 14:
A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a machine learning specialist 
will build a binary classifier based on two features: age of account, denoted by x, and transaction month, denoted by y. The class 
distributions are illustrated in the provided figure. The positive class is portrayed in red, while the negative class is portrayed 
in black. Which model would have the HIGHEST accuracy?

    Diagram: 
      - shows a red square surrounded on all sides with a sea of black dots that reduce in density with distance from the red square

    Long short-term memory (LSTM) model with scaled exponential linear unit (SELU).
    Logistic Regression.
  x Support vector machine (SVM) with non-linear kernel.
    Single perceptron with tanh activation function.



-------------------

        Confusion Matrix:

                        |  Predicted Class     |     Predicted Class
                        |  Negative            |     Positive
                        |  (NOT FRAUD)         |     (FRAUD)
          --------------|----------------------|----------------------
          Actual Class  | True Negative (TN)   |  False Positive (FP)       
                        |                      |                             
          Negative      | NOT FRAUD was        | FRAUD was incorrectly      
          (NOT FRAUD)   | correctly predicted  | predicted as NOT FRAUD    ^
                        | as NOT FRAUD         |                           |
          --------------|----------------------|----------------------     |
          Actual Class  | False Negative (FN)  |  True Positive (TP)       |
                        |                      |                           |
          Postive       | FRAUD was incorrectly| FRAUD was correctly       |
          (FRAUD)       | predicted as         | predicted as FRAUD       Precision
                        | NOT FRAUD            |                             
          --------------|----------------------|----------------------      
                                                  <-------- Recall

        Metric for Classification Problems

           Accuracy: (TP + TN)  / (TP + FP + TN + FN)
              - percentage of predictions that were correct:
              - less effective with a lot of true negatives
                 - example: predicting fraud with little to no fraud data

           Precision: (TP)  / (TP + FP)
              - accuracy of positive predictions
              - percentage of positive predictions that were correct:
              - Use when the cost of false positives is high
                 - example: an email is flagged and deleted as spam when it really isn't

           Recall: (TP)  / (TP + FN)
              - also called sensitivity or true positive rate (TPR)
              - percentage of actual positive predictions that were correctly identified:
              - Use when the cost of false negatives is high
                 - example: someone has cancer, but screening does not find it

           F1 Score: (TP)  / [TP + ((FN + FP) / 2)]
             - combined precision and recall score
             - harmonic mean of the precision and recall 
             - regular mean treats all values equally, the harmonic mean give more weight to low values
             - classifiers will only get high F1 Score if both recall and precision values are high

           Equation 3-3: F1 score:

           F1 = 2 / [ (1/precision) + (1/recall)]  =  2 x [( precision x recall) / (precision + recall)] 

              = (TP)  / [TP + ((FN + FP) / 2)]


-------------------
O'Reilly Hands-ON Machine Learning with Scikit-Learn, Keras, and TensorFlow

4.3 Logistic Regressions (pages 164 - 173)

    Logistic Regressions 
      - logistic regressions (also called logit regression) is used to estimate the probability that an
        instance belongs to a particular class 
      - if the probability estimate is greater than the threshold (typicall 50%), then it predicts the
        instance belongs to the class (called 'positive label')
      - if the probability estimate is less than the threshold (typicall 50%), then it predicts the
        instance does not belong to the class (called 'negative label')
      - this is a 'binary classifier'

  Estimating Probabilities (pages 164 - 165)

    logistic regression model
      - computes the weighted sum of the input features (plus a bias term), but instead of outputing the 
        results directly like the linear regression model does, it outputs the 'logistic of this result'

     Equation 4-13 Logistic regression model estimated probability (vectorized form)

        pred-p = h-theta(x) = sigma(theta-transpose * x)


     logistic function:
        - a sigmoid function (i.e. S-shared) that a number between 0 and 1
        - it is defined in Equation 4-14

     Equation 4-14 Logistic Function

        sigma(t) = 1 / (1 + exp(-t))

           t: score
         
          where exp:  e exponential function

      logit:
        - the score 't' is often called the 'logit'
        - the name comes from the logit function, defined as logit(p) = log(p/(1 - p)),
          is the inversed of the logistic function
        - if you compute the 'logit' of the estimated probability 'p', you will find the result is 't'

-------------------
O'Reilly Hands-ON Machine Learning with Scikit-Learn, Keras, and TensorFlow


Chapter 5 exercises:

1. What is the fundamental idea behind support vector machines?

  -> identify the widest possible 'street' (hyperplane) between 2 classes 

  book answer: The fundamental idea behind Support Vector Machines is to fit the widest possible "street" between 
  the classes. In other words, the goal is to have the largest possible margin between the decision boundary that 
  separates the two classes and the training instances. When performing soft margin classification, the SVM searches 
  for a compromise between perfectly separating the two classes and having the widest possible street (i.e., a few 
  instances may end up on the street). Another key idea is to use kernels when training on nonlinear datasets. SVMs 
  can also be tweaked to perform linear and nonlinear regression, as well as novelty detection.

2. What is a support vector?

   -> the instances located on street (hyperplane separating classes) including instances located on the 
   edge of the street

  book answer: After training an SVM, a support vector is any instance located on the "street" (see the previous 
  answer), including its border. The decision boundary is entirely determined by the support vectors. Any instance 
  that is not a support vector (i.e., is off the street) has no influence whatsoever; you could remove them, add 
  more instances, or move them around, and as long as they stay off the street they won't affect the decision 
  boundary. Computing the predictions with a kernelized SVM only involves the support vectors, not the whole training set.



-------------------


Guide on Support Vector Machine (SVM) Algorithm
  https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/

What is a Support Vector Machine(SVM)?
  - It is a supervised machine learning problem where we try to find a hyperplane that best separates the two classes.

  - Note: Don’t get confused between SVM and logistic regression. Both the algorithms try to find the best hyperplane, but the main 
          difference is logistic regression is a probabilistic approach whereas support vector machine is based on statistical approaches.

Now the question is which hyperplane does it select?  
 - SVM does this by finding the maximum margin between the hyperplanes that means maximum distances between the two classes.


Logistic Regression vs Support Vector Machine (SVM)

  - Depending on the number of features you have you can either choose Logistic Regression or SVM.
  - SVM works best when the dataset is small and complex. 
  - It is usually advisable to first use logistic regression and see how does it performs, if it fails to give a good accuracy 
    you can go for SVM without any kernel (will talk more about kernels in the later section). 
  - Logistic regression and SVM without any kernel have similar performance but depending on your features, one may be 
    more efficient than the other.


Kernels in Support Vector Machine

  - The most interesting feature of SVM is that it can even work with a non-linear dataset and for this, we use 
    “Kernel Trick” which makes it easier to classifies the points. 

  - Used When you cannot draw a single line or say hyperplane which can classify the points correctly. 
  - what we do is try converting this lower dimension space to a higher dimension space using some quadratic functions which 
    will allow us to find a decision boundary that clearly divides the data points. 
  - These functions which help us do this are called Kernels and which kernel to use is purely determined by hyperparameter tuning. 

RBF Kernel

  - What it actually does is to create non-linear combinations of our features to lift your samples onto a higher-dimensional 
    feature space where we can use a linear decision boundary to separate your classes 
  - It is the most used kernel in SVM classifications, the following formula explains it mathematically:

  Formula for RBF kernal
     K(x1, x2) = exp (-||X1 - X2||**2 / 2 * sigma**2)

     where,
     1. ‘sigma’ is the variance and our hyperparameter 
     2. ||X1 - X2|| is the Euclidean (L2-norm) Distance between two points X1 and X2

SVM in Machine Learning Summary:

  - A popular and reliable supervised machine learning technique called Support Vector Machine (SVM) was first created for 
    classification tasks, though it can also be modified to solve regression issues. 
  - The goal of SVM is to locate in the feature space the optimal separation hyperplane between classes.

  Important Ideas of SVM Hyperplane: 
    - The feature space’s decision border dividing several classes. 
    - This becomes a flat affine subspace in higher dimensions, however in two dimensions it would be a line.

  Margin: 
    - The separation of any class’s closest data points from the hyperplane. 
    - SVM makes the most of this leeway to guarantee the greatest gap between classes. 
    - Before making any mistakes, the goal is to make the “street” between the classes as wide as possible.

  Support vectors 
    - Support Vectors are the data points that are closest to the hyperplane and play a crucial role in determining the 
      hyperplane’s location and orientation. 
    - These locations, known as support vectors, have a direct impact on the ideal hyperplane.

-------------------

Radial Basis Function (RBF) Kernel (Note: excellent RBF description):
https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a

  The RBF kernel function for two points X1 and X2 computes the similarity or how close they are to each other. 
  This kernel can be mathematically represented as follows:
 
     K(x1, x2) = exp (-||X1 - X2||**2 / 2 * sigma**2)
  where,
  1. ‘sigma’ is the variance and our hyperparameter 
  2. ||X1 - X2|| is the Euclidean (L2-norm) Distance between two points X1 and X2

  The maximum value that the RBF kernel can be is 1 and occurs when d12 (distance between x1 and x2)
  is 0 which is when the points are the same, i.e. X1 = X2  (maximum similarity).

-------------------------------------------------------------------------------------
Question 18:
A Machine Learning Specialist is packaging a custom ResNet model into a Docker container so the company can leverage Amazon SageMaker 
for training. The Specialist is using Amazon EC2 P3 instances to train the model and needs to properly configure the Docker container 
to leverage the NVIDIA GPUs. What does the Specialist need to do?

    Bundle the NVIDIA drivers with the Docker image.
  x Build the Docker container to be NVIDIA-Docker compatible.
    Organize the Docker container's file structure to execute on GPU instances.
    Set the GPU flag in the Amazon SageMaker CreateTrainingJob request body.



-------------------
CUDA (Compute Unified Device Architecture) Zone
  https://developer.nvidia.com/cuda-zone

  - CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical 
    processing units (GPUs).
  - The CUDA Toolkit from NVIDIA provides everything you need to develop GPU-accelerated applications. 
  - The CUDA Toolkit includes GPU-accelerated libraries, a compiler, development tools and the CUDA runtime.

  - When using CUDA, developers program in popular languages such as C, C++, Fortran, Python and MATLAB and express 
    parallelism through extensions in the form of a few basic keywords.
  
-------------------
Custom Docker containers with SageMaker
  https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-adapt-your-own.html

  There are two toolkits that allow you to bring your own container and adapt it to work with SageMaker:

    SageMaker Training Toolkit
      – Use this toolkit for training models with SageMaker.

    SageMaker Inference Toolkit
     – Use this toolkit for deploying models with SageMaker.

-------------------
How Amazon SageMaker Runs Your Training Image
  https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html
  - If you plan to use GPU devices for model training, make sure that your containers are nvidia-docker compatible. 
  - Include only the CUDA toolkit on containers; don't bundle NVIDIA drivers with the image. 
  - For more information about nvidia-docker, see NVIDIA/nvidia-docker.

-------------------
NVIDIA Container Toolkit

  https://github.com/NVIDIA/nvidia-container-toolkit

  Introduction
    - The NVIDIA Container Toolkit allows users to build and run GPU accelerated containers. 
    - The toolkit includes a container runtime library and utilities to automatically configure containers to leverage NVIDIA GPUs.

-------------------------------------------------------------------------------------
Question 19:
A Machine Learning Specialist is building a Logistic Regression model that will predict whether or not a person will order a pizza. 
The Specialist is trying to build the optimal model with an ideal classification threshold. What model evaluation technique should 
the Specialist use to understand how different classification thresholds will impact the model's performance?

  x Receiver operating characteristic (ROC) curve.
    Misclassification rate.
    Root Mean Square Error (RMSE).
    L1 norm.

-------------------
Interpreting Logistic ROC Curves
  https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_logistic_roc_curves.htm

  - ROC curves in logistic regression are used for determining the best cutoff value for predicting whether a new observation 
    is a "failure" (0) or a "success" (1)

-------------------

O'Reilly Hands-ON Machine Learning with Scikit-Learn, Keras, and TensorFlow

The ROC Curve:

  The receiver operating characteristic (ROC) curve plots the True Positive rate (TPR) (aka recall) against 
    the False Positive Rate (FPR).
  The FPR (aka fall-out) is the ratio of negative instances incorrectly classified as positive.
  The TNR (true negative rate) (aka specificity) is the ration of negative instances that are correctly classified as negative. 
   
           ROC Curve: plots  TPR (aka recall)  versus   FPR 
                      -> equivalent: sensitivity (recall) versus 1 - specificity

           TPR or Recall or sensitivity:       TP  / (TP + FN)

           TNR or specificity:                 TN / (FP + TN)   

           FPR (or fall-out):                  FP  / (FP + TN) = 1 - TNR

           FNR              :                  FN  / (TP + FN) 

           FPR              :                  FP  / (FP + TN)


-------------------

Classification: ROC and AUC
  https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

  - The ROC curve is a visual representation of model performance across all thresholds. 
  - The long version of the name, receiver operating characteristic, is a holdover from WWII radar detection.

  - The ROC curve is drawn by calculating the true positive rate (TPR) and false positive rate (FPR) at every 
    possible threshold (in practice, at selected intervals), then graphing TPR over FPR. 
  - A perfect model, which at some threshold has a TPR of 1.0 and a FPR of 0.0, can be represented by either a 
    point at (0, 1) if all other thresholds are ignored

  Area under the curve (AUC)
    - The area under the ROC curve (AUC) represents the probability that the model, if given a randomly chosen positive 
      and negative example, will rank the positive higher than the negative.

    - AUC is a useful measure for comparing the performance of two different models, as long as the dataset is roughly balanced. 
    - (See Precision-recall curve, above, for imbalanced datasets.) 
    - The model with greater area under the curve is generally the better one.

Notes:
     
      RoC curve:
         TPR - y-axis      = TP / (TP + FN)
         FPR - x-axis      = FP / (FP + TN)


-------------------
Precision-Recall Curve | ML
  https://www.geeksforgeeks.org/precision-recall-curve-ml/


  Precision-Recall (PR) Curve in Machine Learning 
    - A PR curve is simply a graph with Precision values on the y-axis and Recall values on the x-axis. 
    - In other words, the PR curve contains TP/(TP+FP) on the y-axis and TP/(TP+FN) on the x-axis.
      - It is important to note that Precision is also called the Positive Predictive Value (PPV).
      - The recall is also called Sensitivity, Hit Rate, or True Positive Rate (TPR).

  When to Use ROC vs. Precision-Recall Curves?
    - ROC (Receiver Operating Characteristic) curves are suitable when the class distribution is balanced, and false positives 
      and false negatives have similar consequences. 
    - They depict the trade-off between sensitivity and specificity. 

    - In contrast, Precision-Recall curves are preferable when dealing with imbalanced datasets, focusing on positive 
      class prediction performance.
    - Precision-Recall provides insights into the model’s ability to correctly classify positive instances. 

-------------------------------------------------------------------------------------
Quesion 23:
A Machine Learning Specialist is building a Convolutional Neural Network (CNN) that will classify 10 types of animals. The Specialist 
has built a series of layers in a neural network that will take an input image of an animal, pass it through a series of convolutional 
and pooling layers, and then finally pass it through a dense and fully connected layer with 10 nodes. The Specialist would like to get 
an output from the neural network that is a probability distribution of how likely it is that the input image belongs to each of the 
10 classes. Which function will produce the desired output?

    Dropout.
    Smooth L1 loss.
  x Softmax.
    Rectified linear units (ReLU).

-------------------

      https://www.pinecone.io/learn/softmax-activation/
       - The softmax activation function transforms the raw outputs of the neural network into a vector of probabilities, 
         essentially a probability distribution over the input classes. 
       - Consider a multiclass classification problem with N classes. 
          - The softmax activation returns an output vector that is N entries long, with the entry at index i corresponding 
            to the probability of a particular input belonging to the class i.
       - proceed to understand why you cannot use the sigmoid or argmax activations in the output layer for multiclass 
         classification problems.
       Softmax function:
         - you can think of the softmax function as a vector generalization of the sigmoid activation
         - The softmax activation function takes in a vector of raw outputs of the neural network and returns a vector of probability scores.
         - All entries in the softmax output vector are between 0 and 1.
         - In a multiclass classification problem, where the classes are mutually exclusive, notice how the entries of the softmax 
           output sum up to 1: 0.664 + 0.249 + 0.087 = 1. (see example in above website)


        equation of the softmax function
            softmax(z)_i = exp (z_i)) / [ SUM_j exp(z_j)) ]
                  where:
                    - SUM_j is from j = 1  to j = N
                    - z is the vector of raw outputs from the neural network
                    - The value of e ~= 2.718
                    - The i-th entry in the softmax output vector softmax(z) can be thought of as the predicted probability 
                      of the test input belonging to class i.
                    - N: number of classes 

-------------------
https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

Activation Functions

    Linear
    ELU
    ReLU
    LeakyReLU
    Sigmoid
    Tanh
    Softmax

  ReLU (Rectified Linear Units)
    - A recent invention which stands for Rectified Linear Units. 
    - The formula is deceptively simple: max(0,z). 
    - Despite its name and appearance, it’s not linear and provides the same benefits as Sigmoid (i.e. the ability to 
      learn nonlinear functions), but with better performance.

   Pros
     - It avoids and rectifies vanishing gradient problem.
     - ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations.

   Cons

     - One of its limitations is that it should only be used within hidden layers of a neural network model.
     - . . .

  Softmax
    - Softmax function calculates the probabilities distribution of the event over ‘n’ different events. 
    - In general way of saying, this function will calculate the probabilities of each target class over all possible target classes. 
    - Later the calculated probabilities will be helpful for determining the target class for the given inputs.

-------------------------------------------------------------------------------------
Quesion 24:
A Machine Learning Specialist trained a regression model, but the first iteration needs optimizing. The Specialist needs to 
understand whether the model is more frequently overestimating or underestimating the target. What option can the Specialist 
use to determine whether it is overestimating or underestimating the target value?

    Root Mean Square Error (RMSE).
  x Residual plots.
    Area under the curve.
    Confusion matrix.

-------------------

Residual: 
  - A residual is the vertical difference between the actual value and the predicted value. That is,

        residual = actual_y - predicted_y  = y - y_cap

Residual Plot: 
  - A residual plot is a scatterplot that displays the residuals on the vertical axis and the independent variable 
    on the horizontal axis. 
  - Residual plots help us to determine whether a linear model is appropriate in modeling the given data.

  - Since a residual is the "leftover" value after subtracting the expected value from the actual value and the expected 
    value is obtained through a a linear model such as a line of best fit, a residual plot shows how the data points 
    deviate from the model.

  - If the residuals are randomly scattered around the residual = 0, it means that a linear model approximates the data 
    points well without favoring certain inputs. In such a case, we conclude that a linear model is appropriate.

  - If the residuals show a curved pattern, it indicates that a linear model captures the trend of some data points better 
    than that of others. In such a case, we should consider using a model other than a linear model. 

-------------------------------------------------------------------------------------
Question 25:
A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a Machine Learning 
Specialist would like to build a binary classifier based on two features: age of account and transaction month. The class 
distribution for these features is illustrated in the figure provided. Based on this information, which model would have 
the HIGHEST recall with respect to the fraudulent class?

  Notes: 
       - the figure shows fraud generally occurs when age of account is between 3 and 7 and transaction month is
         is between 3 and 7 suggesting these features are related (not independent)
       - fradulent transactions are in sphere bounded by 3 and 7 transaction month (x-axis) and account age is between 3 and 7;
         normal transaction surround these sphere

    Decision tree.
    Linear support vector machine (SVM).
  x Naive Bayesian classifier.
    Single Perceptron with sigmoidal activation function.


-------------------

  Naive Bayes Classifiers
    https://www.geeksforgeeks.org/naive-bayes-classifiers/

   - This model predicts the probability of an instance belongs to a class with a given set of feature value. 
   - It is a probabilistic classifier. 
   - It is because it assumes that one feature in the model is independent of existence of another feature. 
   - In other words, each feature contributes to the predictions with no relation between each other. In real world, this 
     condition satisfies rarely. It uses Bayes theorem in the algorithm for training and prediction

   - my short form take: it is the sum of the probabililty for each feature based on the value of the feature

   today = (Sunny, Hot, Normal, False)

     Probability to play golf is given by:
        P(Yes|today)=P(SunnyOutlook|Yes) P(HotTemperature|Yes) P(NormalHumidity|Yes) P(NoWind|Yes) P(Yes) / P(today)

    probability to not play golf is given by:
       P(No|today)= P(SunnyOutlook|No) P(HotTemperature|No) P(NormalHumidity|No) P(NoWind|No) P(No) / P(today)

    Since, P(today) is common in both probabilities, we can ignore P(today) and find proportional probabilities as:


    Best Quess: Since these two features appear to be related, the recall will likely by high?

-------------------------------------------------------------------------------------
Question 28:
A company is using Amazon Polly to translate plaintext documents to speech for automated company announcements. However, 
company acronyms are being mispronounced in the current documents. How should a Machine Learning Specialist address this 
issue for future documents?

    Convert current documents to SSML with pronunciation tags.
  x Create an appropriate pronunciation lexicon.
    Output speech marks to guide in pronunciation.
    Use Amazon Lex to preprocess the text files for pronunciation.

-------------------
AWS -> Documentation -> Amazon Polly -> Developer Guide -> Managing lexicons
  https://docs.aws.amazon.com/polly/latest/dg/managing-lexicons.html

  - Pronunciation lexicons enable you to customize the pronunciation of words.   
  - Amazon Polly provides API operations that you can use to store lexicons in an AWS region. 
  - Those lexicons are then specific to that particular region. You can use one or more of the lexicons from that region 
    when synthesizing the text by using the SynthesizeSpeech operation. 
  - This applies the specified lexicon to the input text before the synthesis begins.

Note
  - These lexicons must conform with the Pronunciation Lexicon Specification (PLS) W3C recommendation. 
  - For more information, see Pronunciation Lexicon Specification (PLS) Version 1.0 on the W3C website
    https://www.w3.org/TR/pronunciation-lexicon/
-------------------
AWS -> Documentation -> Amazon Polly -> Developer Guide -> Generating speech from SSML documents
  https://docs.aws.amazon.com/polly/latest/dg/ssml.html

  - You can use Amazon Polly to generate speech from either plain text or from documents marked up with Speech Synthesis Markup Language (SSML). 
  - Using SSML-enhanced text gives you additional control over how Amazon Polly generates speech from the text you provide.

  - When using SSML, you enclose the entire text in a <speak> tag to let Amazon Polly know that you're using SSML. For example:
    <speak>Hi! My name is Joanna. I will read any text you type here.</speak>

  - For example, you can include a long pause within your text, or change the speech rate or pitch. Other options include:
      - emphasizing specific words or phrases
      - using phonetic pronunciation
      - including breathing sounds
      - whispering
      - using the Newscaster speaking style.

-------------------------------------------------------------------------------------
Question 30:
When submitting Amazon SageMaker training jobs using one of the built-in algorithms, which common parameters MUST be specified? (Choose three.)

    The training channel identifying the location of training data on an Amazon S3 bucket.
    The validation channel identifying the location of validation data on an Amazon S3 bucket.
  x The IAM role that Amazon SageMaker can assume to perform tasks on behalf of the users.
    Hyperparameters in a JSON array as documented for the algorithm used.
  X The Amazon EC2 instance class specifying whether training will be run using CPU or GP
  X The output path specifying where on an Amazon S3 bucket the trained model will persist.

-------------------
https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html
CreateTrainingJob

  - Starts a model training job. After training completes, SageMaker saves the resulting model artifacts to an Amazon S3 location that you specify. 
  Required Parameters:

   AlgorithmSpecification
     - The registry path of the Docker image that contains the training algorithm and algorithm-specific metadata, including the input mode

    OutputDataConfig
      - Specifies the path to the S3 location where you want to store model artifacts. SageMaker creates subfolders for the artifacts. 

    ResourceConfig
      - The resources, including the ML compute instances and ML storage volumes, to use for model training. 

    RoleArn
      - The Amazon Resource Name (ARN) of an IAM role that SageMaker can assume to perform tasks on your behalf. 

    StoppingCondition
      - Specifies a limit to how long a model training job can run. It also specifies how long a managed Spot training job has to complete.
      Note: May only be required when using Spot instances?

    TrainingJobName
       - The name of the training job. The name must be unique within an AWS Region in an AWS account. 
-------------------

         >>> from sagemaker import image_uris
         >>> container = image_uris.retrieve('xgboost', boto3.Session().region_name, '1')
             
         >>> # Next, because we're training with the CSV file format, we'll create inputs that our training function can use as a pointer 
         >>> #  to the files in S3, which also specify that the content type is CSV.
         >>> 
         >>> s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/algorithms_lab/xgboost_train'.format(bucket), content_type='csv')
         >>> s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://{}/algorithms_lab/xgboost_validation'.format(bucket), content_type='csv')

         >>> # Create a training job name
         >>> job_name = 'ufo-xgboost-job-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
         >>> print('Here is the job name {}'.format(job_name))
         >>> 
         >>> # Here is where the model artifact will be stored
         >>> output_location = 's3://{}/algorithms_lab/xgboost_output'.format(bucket

         >>> sess = sagemaker.Session()
         >>> 
         >>> xgb = sagemaker.estimator.Estimator(container, role, instance_count=1, instance_type='ml.m4.xlarge',
         >>>                                     output_path=output_location, sagemaker_session=sess)
         >>> 
         >>> xgb.set_hyperparameters(objective='multi:softmax', num_class=3, num_round=100)
         >>> 
         >>> data_channels = { 'train': s3_input_train, 'validation': s3_input_validation }
         >>> xgb.fit(data_channels, job_name=job_name) 


-------------------------------------------------------------------------------------


Question 37:
Which characteristic applies to a catalog backup?

    Catalog staging files deleted after a successful catalog backup.
 x  A catalog backup can be configured to send disaster recovery information to an e-mail address.      
    A catalog backup must fit on a single tape.
    A catalog backup shuts down the NetBackup database.

Note: Assume "Catalog" is AWS Glue Catalog. AWS Glue Catalog does not appear to have a backup mechanism

-------------------
AWS Glue Data Catalog
https://docs.aws.amazon.com/prescriptive-guidance/latest/serverless-etl-aws-glue/aws-glue-data-catalog.html

The AWS Glue Data Catalog is a centralized metadata repository for all your data assets across various data sources. It provides 
a unified interface to store and query information about data formats, schemas, and sources. When an AWS Glue ETL job runs, it uses 
this catalog to understand information about the data and ensure that it is transformed correctly.

The AWS Glue Data Catalog is composed of the following components:

    Databases and tables

    Crawlers and classifiers

    Connections

    Schema Registry


AWS Glue Schema Registry
  - The AWS Glue Schema Registry provides a centralized location for managing and enforcing data stream schemas. 
  - It enables disparate systems, such as data producers and consumers, to share a schema for serialization and deserialization. 
  - Sharing a schema helps these systems to communicate effectively and avoid errors during transformation.

  - The Schema Registry ensures that downstream data consumers can handle changes made upstream, because they are aware of the expected schema. 
  - It supports schema evolution, so that a schema can change over time while maintaining compatibility with previous versions of the schema.

  - The Schema Registry integrates with many AWS services, including Amazon Kinesis Data Streams, Firehose, and Amazon Managed Streaming for 
    Apache Kafka. 
 
-------------------------------------------------------------------------------------
Question 38:
A data scientist is developing a pipeline to ingest streaming web traffic data. The data scientist needs to implement a process to 
identify unusual web traffic patterns as part of the pipeline. The patterns will be used downstream for alerting and incident response. 
The data scientist has access to unlabeled historic data to use, if needed. The solution needs to do the following: Calculate an anomaly 
score for each web traffic entry. Adapt unusual event identification to changing web patterns over time. Which approach should the 
data scientist implement to meet these requirements?

    Use historic web traffic data to train an anomaly detection model using the Amazon SageMaker Random Cut Forest (RCF) built-in model. 
    Use an Amazon Kinesis Data Stream to process the incoming web traffic data. Attach a preprocessing AWS Lambda function to perform data 
    enrichment by calling the RCF model to calculate the anomaly score for each record.

    Use historic web traffic data to train an anomaly detection model using the Amazon SageMaker built-in XGBoost model. Use an Amazon 
    Kinesis Data Stream to process the incoming web traffic data. Attach a preprocessing AWSLambda function to perform data enrichment by 
    calling the XGBoost model to calculate the anomaly score for each record.

    Collect the streaming data using Amazon Kinesis Data Firehose. Map the delivery stream as an input source for Amazon Kinesis Data Analytics. 
    Write a SQL query to run in real time against the streaming data with the k-NearestNeighbors (kNN) SQL extension to calculate anomaly 
    scores for each record using a tumbling window.

  x Collect the streaming data using Amazon Kinesis Data Firehose. Map the delivery stream as an input source for Amazon Kinesis Data Analytics. 
    Write a SQL query to run in real time against the streaming data with the Amazon RandomCut Forest (RCF) SQL extension to calculate 
    anomaly scores for each record using a sliding window.

-------------------

  AWS Kinesis Data Analytics - RANDOM_CUT_FOREST
  https://docs.aws.amazon.com/kinesisanalytics/latest/sqlref/sqlrf-random-cut-forest.html

  Detects anomalies in your data stream.  A record is an anomaly if it is distant from other records.  To detect anomalies in individual 
  record columns, see RANDOM_CUT_FOREST_WITH_EXPLANATION.

  A stream record can have non-numeric columns, but the function uses only numeric columns to assign an anomaly score. A record can have one 
  or more numeric columns. The algorithm uses all of the numeric data in computing an anomaly score.  If a record has n numeric columns, 
  the underlying algorithm assumes each record is a point in n-dimensional space. A point in n-dimensional space that is distant from 
  other points receives a higher anomaly score. 

-------------------------------------------------------------------------------------
Question 41:
A Machine Learning Specialist is assigned to a Fraud Detection team and must tune an XGBoost model, which is working appropriately 
for test data. However, with unknown data, it is not working as expected. The existing parameters are provided as follows. Which 
parameter tuning guidelines should the Specialist follow to avoid overfitting?

    param = {
       'eta'=0.05,        # the training step for each iteration
       'silent'=3,        # logging mode - quiet
       'n_estimators': 2000,
       'max_depth': 30,
       'min_child_weight': 3,
       'gamma': 0,
       'subsample': 0.8,
       'objective':"multi:softprob",   # error evaluation for multiclass training
       'num_class':201,                # number of classes that exist in this dataset    
       'num_round': 60                 # the number of training iterations
       }


    Increase the max_depth parameter value.
  x Lower the max_depth parameter value.
    Update the objective to binary:logistic.
    Lower the min_child_weight parameter value.

-------------------

   'eta'=0.05,     
   'silent'=3,    
   'n_estimators': 2000,
      -> not a XGBoost hyperparmeter
   'max_depth' : 30 	
      - Maximum depth of a tree. Increasing this value makes the model more complex and likely to be overfit. 0 indicates no limit. 
        A limit is required when grow_policy=depth-wise.
      - Optional;  Valid values: Integer. Range: [0,∞);  Default value: 6
   'min_child_weight': 3,
      - Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with 
        the sum of instance weight less than min_child_weight, the building process gives up further partitioning. In linear 
        regression models, this simply corresponds to a minimum number of instances needed in each node. The larger the algorithm, 
        the more conservative it is.
      - Optional; Valid values: Float. Range: [0,∞).; Default value: 1
   'gamma': 0,
     - Minimum loss reduction required to make a further partition on a leaf node of the tree. 
     - The larger, the more conservative the algorithm is.
     - Optional; Valid values: Float. Range: [0,∞).; Default value: 0
   'subsample': 0.8,
     - Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collects half of the data 
       instances to grow trees. This prevents overfitting.
     - Optional; Valid values: Float. Range: [0,1]. ; Default value: 1
   'objective':"multi:softprob", 
     - Specifies the learning task and the corresponding learning objective. Examples: reg:logistic, multi:softmax, 
       reg:squarederror. For a full list of valid inputs, refer to XGBoost Learning Task Parameters
     - Optional;  Valid values: String;  Default value: "reg:squarederror"
   'num_class':201,             
     - The number of classes.
     - Required if objective is set to multi:softmax or multi:softprob.; Valid values: Integer.
   'num_round': 60             
     - The number of rounds to run the training.
     - Required; Valid values: Integer.

-------------------

https://stats.stackexchange.com/questions/443259/how-to-avoid-overfitting-in-xgboost-model

XGBoost (and other gradient boosting machine routines too) has a number of parameters that can be tuned to avoid over-fitting. 
I will mention some of the most obvious ones. For example we can change:

  colsample_bytree
    - the ratio of features used (i.e. columns used);. 
    - Lower ratios avoid over-fitting.
  subsample
    - the ratio of the training instances used (i.e. rows used); 
    - Lower ratios avoid over-fitting.
  max_depth
    - the maximum depth of a tree; 
    - Lower values avoid over-fitting.
  gamma
    - the minimum loss reduction required to make a further split; 
    - Larger values avoid over-fitting.
  eta
    - the learning rate of our GBM (i.e. how much we update our prediction with each successive tree); 
    - Lower values avoid over-fitting.
  min_child_weight
    - the minimum sum of instance weight needed in a leaf, in certain applications this relates directly to the minimum 
      number of instances needed in a node; 
    - Larger values avoid over-fitting.

-------------------------------------------------------------------------------------
Question 43:
A Machine Learning Specialist is working for an online retailer that wants to run analytics on every customer visit, processed 
through a machine learning pipeline. The data needs to be ingested by Amazon Kinesis Data Streams at up to 100 transactions 
per second, and the JSON data blob is 100 KB in size. What is the MINIMUM number of shards in Kinesis Data Streams the Specialist 
should use to successfully ingest this data?

    1 shards.
 x  10 shards.
    100 shards.
    1,000 shards.

Note:  100 transactions/sec x 100KB/transaction = 10MB/sec and max total write rate is 1MB per sec ->  10MB / 1Mb = 10 shards
-------------------

  Kinesis Shards
    - only applies to Kinesis data streams
    - Kinesis streams are made up of Shards
    - each shard is a sequence of one or more data records and provides a fixed unit of capacity
       - 5 read transactions per second
       - The max total read rate is 2MB per second
       - 1K records writes per second
       - The max total write rate is 1MB per second
    -  Data records are composed of a sequence number, a partition key, and a data blob, which is an immutable sequence of bytes

-------------------------------------------------------------------------------------
Question 44:
A Machine Learning Specialist is deciding between building a naive Bayesian model or a full Bayesian network for a classification 
problem. The Specialist computes the Pearson correlation coefficients between each feature and finds that their absolute values 
range between 0.1 to 0.95. Which model describes the underlying data in this situation?

    A naive Bayesian model, since the features are all conditionally independent.
    A full Bayesian network, since the features are all conditionally independent.
    A naive Bayesian model, since some of the features are statistically dependent.
  x A full Bayesian network, since some of the features are statistically dependent.

-------------------

Pearson's Correlation
  https://www.sciencedirect.com/topics/computer-science/pearson-correlation

  - Similarity scores are based on comparing one data object with another, attribute by attribute, usually summing the squares of the 
    differences in magnitude for each attribute, and using the calculation to compute a final outcome, known as the correlation score. 
  - One of the most popular correlation methods is Pearson's correlation, which produces a score that can vary from − 1 to + 1. 
    - Two objects with a high score (near + 1) are highly similar.
    - Two uncorrelated objects would have a Pearson score near zero. 
    - Two objects that correlated inversely (ie, one falling when the other rises) would have a Pearson score near − 1 

   - The Pearson correlation for two objects, with paired attributes, sums the product of their differences from their object means, 
     and divides the sum by the product of the squared differences from the object means (Fig. 4.15).

            ∑[(x_i - X_mean)(y_i - y_mean)] 
            ------------------------------
           √∑(x_i - X_mean)**2  √∑(y_i - y_mean)**2 

Difference between Naive Bayes and Full Bayesian Network Model
  https://medium.com/@mansi89mahi/simple-explanation-difference-between-naive-bayes-and-full-bayesian-network-model-505616545503

  The Naive Bayes model and a full Bayesian network are both probabilistic models used in machine learning and statistics, 
  but they have significant differences in terms of complexity and modeling assumptions.

    Naive Bayes Model:

      Simplicity: 
        - The Naive Bayes model is one of the simplest probabilistic classifiers. 
        - It’s based on Bayes’ theorem and assumes that all features are conditionally independent given the class label. 
          This is a strong and often unrealistic assumption, which is why it’s called “naive.”

       Use Case: 
         - Naive Bayes is commonly used for text classification tasks, spam detection, and other simple classification problems 
           where the assumption of feature independence is not too far from reality.

       Efficiency: 
         - Due to its simplicity, Naive Bayes is computationally efficient and can work well with small to moderate-sized datasets.

    Full Bayesian Network:

       Complexity: 
         - A full Bayesian network is a more complex model that allows for dependencies and correlations among variables 
           in a more flexible manner. 
         - It represents a directed acyclic graph (DAG) where nodes correspond to random variables, and edges indicate 
           probabilistic dependencies between variables.

       Conditional Dependencies: 
         - In a full Bayesian network, variables can depend on other variables in a conditional and non-naive way. 
         - This means it can capture complex relationships and dependencies in the data more accurately.

       Use Case: 
         - Bayesian networks are used for modeling complex systems where variables interact in intricate ways, such as medical 
           diagnosis, fault detection in engineering, and decision support systems.

       Complexity and Scalability: 
         - While Bayesian networks can model complex dependencies, they can become computationally expensive as the number of 
           variables and interactions increases. 
         - Inference and learning in large Bayesian networks can be challenging.

-------------------------------------------------------------------------------------
Question 45:
A Data Scientist is building a Linear Regression model and will use resulting p-values to evaluate the statistical significance of 
each coefficient. Upon inspection of the dataset, the Data Scientist discovers that most of the features are normally distributed. 
The plot of one feature in the dataset is shown in the graphic. What transformation should the Data Scientist apply to satisfy the 
statistical assumptions of the Linear Regression model?
   -> plot shows log-normal distribution with a few outliers at end

    Exponential transformation.
  x Logarithmic transformation.
    Polynomial transformation.
    Sinusoidal transformation.

Note: logarthimic transformation of a log-normal distribution results in a normally distribution

-------------------
Log Transformation: Purpose and Interpretation
  https://medium.com/@kyawsawhtoon/log-transformation-purpose-and-interpretation-9444b4b049c9


  Normal Distribution:
    - its mean, median and mode have the same value and it can be defined with just two parameters: mean and variance.
    - it has symmetrical sides with asymptotic tails
      Note: asymptotic means: An asymptotic line is a line that gets closer and closer to a curve as the distance gets 
            closer to infinity.

   Log Transformation:
     - data transformation method in which it replaces each variable x with a log(x). (usually log_e(x) or ln(x)
     - When our original continuous data do not follow the bell curve, we can log transform this data to make it 
       as “normal” as possible so that the statistical analysis results from this data become more valid. 
       - In other words, the log transformation reduces or removes the skewness of our original data. The important 
         caveat here is that the original data has to follow or approximately follow a log-normal distribution. 
         Otherwise, the log transformation won’t work.


Log-normal Distribution - A simple explanation
   https://towardsdatascience.com/log-normal-distribution-a-simple-explanation-7605864fb67c
   Log-normal distribution
      - The log-normal distribution is a right skewed continuous probability distribution, meaning it has a long 
        tail towards the right. 
      - It is used for modelling various natural phenomena such as income distributions, the length of chess games or 
        the time to repair a maintainable system and more.

------

There are four components that explains the statistics of the [linear Regression] coefficients:

    std err stands for Standard Error
    t is the "t-value" of the coefficients
    P>|t| is called the "P-value"
     [0.025  0.975] represents the confidence interval of the coefficients

We will focus on understanding the "P-value" in this module.

The P-value
  - The P-value is a statistical number to conclude if there is a relationship between a feature and the target

  We test if the true value of the coefficient is equal to zero (no relationship). The statistical test for this is called Hypothesis testing.

    - A low P-value (< 0.05) means that the coefficient is likely not to equal zero.
    - A high P-value (> 0.05) means that we cannot conclude that the explanatory variable affects the dependent variable 
      (here: if Average_Pulse affects Calorie_Burnage).
    - A high P-value is also called an insignificant P-value.

Linear Regression P-Value
  - The p-value for each term tests the null hypothesis that the coefficient is equal to zero (no effect). 
  - A low p-value (< 0.05) indicates that you can reject the null hypothesis. In other words, a predictor that has a low p-value 
    is likely to be a meaningful addition to your model because changes in the predictor's value are related to changes in the response variable.
  - Conversely, a larger (insignificant) p-value suggests that changes in the predictor are not associated with changes in the response.

  - In the output below, we can see that the predictor variables of South and North are significant because both of their p-values are 0.000. 
  - However, the p-value for East (0.092) is greater than the common alpha level of 0.05, which indicates that it is not statistically significant.

-------------------------------------------------------------------------------------
Question 49:

  x Augment training data for each item using image variants like inversions and translations, build the model, and iterate.
-------------------

AI generated:

An "augmented image using inversion and translation" refers to a manipulated image where the original picture has been flipped 
vertically (inverted) and then shifted horizontally or vertically (translated) to create a new, modified version of the image, 
often used in data augmentation for machine learning tasks to improve model robustness by exposing it to different variations 
of the same object

-------------------
Image Transformations using OpenCV in Python
  https://www.geeksforgeeks.org/image-transformations-using-opencv-in-python/

  What is Image Transformation?

    Image Transformation involves the transformation of image data in order to retrieve information from the image or preprocess the image for 
    further usage. In this tutorial we are going to implement the following image transformation:

    Image Translation
      Reflection 
      Rotation
      Scaling
      Cropping
      Shearing in x-axis
      Shearing in y-axis

    Image Translation

    - In computer vision or image processing, image translation is the rectilinear shift of an image from one location to another, 
      so the shifting of an object is called translation. 
    - In other words,  translation is the shifting of an object’s location.


-------------------------------------------------------------------------------------
Question 50:
A Data Scientist is developing a binary classifier to predict whether a patient has a particular disease on a series of 
test results. The Data Scientist has data on 400 patients randomly selected from the population. The disease is seen in 
3% of the population. Which cross-validation strategy should the Data Scientist adopt?

    A k-fold cross-validation strategy with k=5.
  x A stratified k-fold cross-validation strategy with k=5.
    A k-fold cross-validation strategy with k=5 and 3 repeats.
    An 80/20 stratified split between training and validation.


-------------------

Stratified sampling 
  - a sampling technique where the samples are selected in the same proportion (by dividing the population into groups called 
    ‘strata’ based on a characteristic) as they appear in the population. 
  - For example, if the population of interest has 30% male and 70% female subjects, then we divide the population into two 
    (‘male’ and ‘female’) groups and choose 30% of the sample from the ‘male’ group and ‘70%’ of the sample from the ‘female’ group.

What is Stratified K-Fold Cross Validation? 
   Stratified k-fold cross-validation is the same as just k-fold cross-validation, But Stratified k-fold 
   cross-validation, it does stratified sampling instead of random sampling.

-------------------------------------------------------------------------------------
Question 53:
A health care company is planning to use neural networks to classify their X-ray images into normal and abnormal classes. 
The labeled data is divided into a training set of 1,000 images and a test set of 200 images. The initial training of a neural network 
model with 50 hidden layers yielded 99% accuracy on the training set, but only 55% accuracy on the test set. What changes should the 
Specialist consider to solve this issue? (Choose three.)

    Choose a higher number of layers.
  X Choose a lower number of layers.
    Choose a smaller learning rate.
  X Enable dropout.
    Include all the images from the test set in the training set.
  X Enable early stopping.

-----------------

Note: Reducing the learning rate should be included as an option for reducing overfitting.


[Medium] The Learning Rate: A Hyperparameter That Matters
  https://mohitmishra786687.medium.com/the-learning-rate-a-hyperparameter-that-matters-b2f3b68324ab

The learning rate is a measure of how much the weights of the neural network are updated each time the model is trained. A higher learning 
rate will cause the weights to be updated more aggressively, while a lower learning rate will cause the weights to be updated more slowly.

A learning rate that is too high can cause the model to overfit the training data.

-------------------------------------------------------------------------------------
Question 56:
A Machine Learning Specialist is given a structured dataset on the shopping habits of a company's customer base. The dataset 
contains thousands of columns of data and hundreds of numerical columns for each customer. The Specialist wants to identify 
whether there are natural groupings for these columns across all customers and visualize the results as quickly as possible. 
What approach should the Specialist take to accomplish these tasks?

 x  Embed the numerical features using the t-distributed stochastic neighbor embedding (t-SNE) algorithm and create a scatter plot.
    Run K-means using the Euclidean distance measure for different values of k and create an elbow plot.
    Embed the numerical features using the t-distributed stochastic neighbor embedding (t-SNE) algorithm and create a line graph.
    Run K-means using the Euclidean distance measure for different values of k and create box plots for each numerical column 
      within each cluster.

-------------------
Introduction to t-SNE
  https://www.datacamp.com/tutorial/introduction-t-sne

  t-distributed Stochastic Neighbor Embedding (t-SNE)
    - t-SNE is an unsupervised non-linear dimensionality reduction technique for data exploration and visualizing high-dimensional data. 
    - Non-linear dimensionality reduction means that the algorithm allows us to separate data that cannot be separated by a straight line.
    - t-SNE gives you a feel and intuition on how data is arranged in higher dimensions. 
    - It is often used to visualize complex datasets into two and three dimensions, allowing us to understand more about underlying 
      patterns and relationships in the data.


  t-SNE vs PCA
    - Both t-SNE and PCA are dimensional reduction techniques that have different mechanisms and work best with different types of data.
    PCA (Principal Component Analysis) 
     - a linear technique that works best with data that has a linear structure. 
     - It seeks to identify the underlying principal components in the data by projecting onto lower dimensions, minimizing variance, 
       and preserving large pairwise distances. 
    t-SNE  
     - a nonlinear technique that focuses on preserving the pairwise similarities between data points in a lower-dimensional space. 
     - t-SNE is concerned with preserving small pairwise distances whereas, PCA focuses on maintaining large pairwise distances 
       to maximize variance.

    - In summary, PCA preserves the variance in the data, whereas t-SNE preserves the relationships between data points in a 
      lower-dimensional space, making it quite a good algorithm for visualizing complex high-dimensional data. 

  How t-SNE works
    - The t-SNE algorithm finds the similarity measure between pairs of instances in higher and lower dimensional space. 
    - After that, it tries to optimize two similarity measures. It does all of that in three steps. 

      1. t-SNE models a point being selected as a neighbor of another point in both higher and lower dimensions. 
        - It starts by calculating a pairwise similarity between all data points in the high-dimensional space using a Gaussian kernel. 
        - The points that are far apart have a lower probability of being picked than the points that are close together. 
      2. Then, the algorithm tries to map higher dimensional data points onto lower dimensional space while preserving the pairwise similarities. 
      3. It is achieved by minimizing the divergence between the probability distribution of the original high-dimensional and lower-dimensional. 
        - The algorithm uses gradient descent to minimize the divergence. The lower-dimensional embedding is optimized to a stable state.

   - The optimization process allows the creation of clusters and sub-clusters of similar data points in the lower-dimensional space 
     that are visualized to understand the structure and relationship in the higher-dimensional data. 

-------------------------------------------------------------------------------------
Question 57:
A Machine Learning Specialist is planning to create a long-running Amazon EMR cluster. The EMR cluster will have 1 master node, 10 core 
nodes, and 20 task nodes. To save on costs, the Specialist will use Spot Instances in the EMR cluster. Which nodes should the Specialist 
launch on Spot Instances?

    Master node.
    Any of the core nodes.
  x Any of the task nodes.
    Both core and task nodes.

-------------------

    EMR cluster architecture
      An EMR cluster is made of 3 node types:
        Primary Node
          - coordinates the distribution of data and tasks
          - tracks status of tasks and monitors health of cluster
        Core Node
          - run tasks and stores data in HDFS (Hadoop Distributed File System).
          - in HDFS, data is stored across multiple instances, with multiple copies
          - Storage is ephemeral (temporary, lasting a short time)
        Task Node
          - (optional) runs tasks but does not store data

-------------------------------------------------------------------------------------
Question 58:
A manufacturer of car engines collects data from cars as they are being driven. The data collected includes timestamp, engine 
temperature, rotations per minute (RPM), and other sensor readings. The company wants to predict when an engine is going to have 
a problem, so it can notify drivers in advance to get engine maintenance. The engine data is loaded into a data lake for training. 
Which is the MOST suitable predictive model that can be deployed into production?

 x  Add labels over time to indicate which engine faults occur at what time in the future to turn this into a supervised learning problem. 
      Use a Recurrent Neural Network (RNN) to train the model to recognize when an engine might need maintenance for a certain fault.
    This data requires an unsupervised learning algorithm. Use Amazon SageMaker K-means to cluster the data.
    Add labels over time to indicate which engine faults occur at what time in the future to turn this into a supervised learning problem. 
      Use a Convolutional Neural Network (CNN) to train the model to recognize when an engine might need maintenance for a certain fault.
    This data is already formulated as a time series. Use Amazon SageMaker seq2seq to model the time series.

-------------------


  CNN vs. RNN: How are they different
  - The main differences between CNNs and RNNs include the following: CNNs are commonly used to solve problems involving spatial 
    data, such as images. 
  - RNNs are better suited to analyzing temporal and sequential data, such as text or videos.


What is a Convolutional Neural Network (CNN)?
  https://medium.com/@hassaanidrees7/ann-vs-cnn-vs-rnn-vs-lstm-understanding-the-differences-in-neural-networks-94486cbb6d5a
  Definition: 
    - CNNs are specialized neural networks designed for processing structured grid data like images. 
    - They use convolutional layers to detect features.
  Structure:
    Convolutional Layers: 
      - Apply filters to the input data to create feature maps.
    Pooling Layers: 
      - Reduce the dimensionality of feature maps.
    Fully Connected Layers: 
      - Connect neurons from previous layers to the output layer.
  Applications:
    - Image and video recognition
    - Object detection
    - Image classification
  Advantages:
    - Excellent at handling spatial data.
    - Requires less preprocessing compared to ANNs.
  Challenges:
    - Computationally intensive.
    - Requires large amounts of labeled data for training.

What is a Recurrent Neural Network (RNN)?
  Definition: 
    - RNNs are designed for sequential data, where each neuron connects to the next layer and to neurons within the same layer.
  Structure:
    - Recurrent Connections: Neurons have connections looping back to themselves, allowing information to persist.
    - Hidden State: Maintains a memory of previous inputs.
  Applications:
    - Time series prediction
    - Natural language processing (NLP)
    - Speech recognition
  Advantages:
    - Effective for sequential and temporal data.
    - Can handle variable-length inputs.
  Challenges:
    - Difficulty in learning long-term dependencies.
    - Prone to vanishing gradient problems.

What is a Long Short-Term Memory Network (LSTM)?
  Definition: 
    - LSTMs are a type of RNN designed to learn long-term dependencies and retain information over longer sequences.
  Structure:
    - Memory Cell: Maintains information over time.
    - Gates: Control the flow of information in and out of the cell (input gate, forget gate, output gate).
  Applications:
    - Long-term time series prediction
    - Advanced NLP tasks (e.g., machine translation)
    - Speech synthesis
  Advantages:
    - Overcomes the vanishing gradient problem.
    - Effective at capturing long-term dependencies.
  Challenges:
    - More complex and computationally intensive than standard RNNs.
    - Requires careful tuning of hyperparameters.

What is an Artificial Neural Network (ANN)?
  Definition: 
    - ANNs are the simplest form of neural networks, consisting of layers of interconnected neurons that process and 
      transmit information.
  Structure:
    - Input Layer: Receives input data.
    - Hidden Layers: Intermediate layers that transform the input data.
    - Output Layer: Produces the final prediction or classification.
  Applications:
    - Basic image recognition
    - Simple pattern recognition
    - Regression tasks
  Advantages:
    - Easy to implement and understand.
    - Suitable for a wide range of problems.
  Challenges:
    - Limited in handling complex data structures.
    - Performance depends on the number of hidden layers and neurons.

Key Differences Between ANN, CNN, RNN, and LSTM
Data Type:

    ANN: General-purpose, works with a variety of data types.
    CNN: Best suited for spatial data (e.g., images).
    RNN: Designed for sequential data (e.g., time series, text).
    LSTM: Enhanced version of RNN, handles long-term dependencies in sequential data.

-------------------------------------------------------------------------------------
Question 62:
A real estate company wants to create a machine learning model for predicting housing prices based on a historical 
dataset. The dataset contains 32 features. Which model will meet the business requirement?

    Logistic Regression.
  x Linear Regression.
    K-means.
    Principal Component Analysis (PCA).

-------------------

ML | Linear Regression vs Logistic Regression
https://www.geeksforgeeks.org/ml-linear-regression-vs-logistic-regression/

  Linear Regression
  - Linear Regression is a machine learning algorithm based on supervised regression algorithm. 
  - Regression models a target prediction value based on independent variables. 
  - It is mostly used for finding out the relationship between variables and forecasting. 
  - Different regression models differ based on – the kind of relationship between the dependent and independent 
    variables, they are considering and the number of independent variables being used. 

  Logistic Regression
    - Logistic regression is basically a supervised classification algorithm. 
    - In a classification problem, the target variable(or output), y, can take only discrete values for a given 
      set of features(or inputs), X.

Sl.No. 	Linear Regression  	                                Logistic Regression
------  --------------------------------------------------      --------------------------------------------------------
1. 	Linear Regression is a supervised regression model. 	Logistic Regression is a supervised classification model
3. 	In Linear Regression, we predict the value by an  	In Logistic Regression, we predict the value by 1 or 0.
         integer number

8. 	It is based on the least square estimation. 	        It is based on maximum likelihood estimation.
10. 	Linear regression is used to estimate the dependent     Whereas logistic regression is used to calculate
        variable in case of a change in independent             the probability of an event. For example, classify if  
        variables. For example, predict the price of houses. 	tissue is benign or malignant. 
-------------------------------------------------------------------------------------
Question 63:
A Machine Learning Specialist is applying a linear least squares regression model to a dataset with 1,000 records and 
50 features. Prior to training, the ML Specialist notices that two features are perfectly linearly dependent. Why could this 
be an issue for the linear least squares regression model?

    It could cause the backpropagation algorithm to fail during training.
  x It could create a singular matrix during optimization, which fails to define a unique solution.
    It could modify the loss function during optimization, causing it to fail during training.
    It could introduce non-linear dependencies within the data, which could invalidate the linear assumptions of the model.

-------------------
Singularity:

  - In regression analysis , singularity is the extreme form of multicollinearity – when a perfect linear relationship exists
    between variables or, in other terms, when the correlation coefficient is equal to 1.0 or -1.0.

  - Such absolute multicollinearity could arise when independent variable are linearly related in their definition. A simple example: 
  - two variables “height in centimeters” and “height in inches” are included in the regression model.

Assumptions in Linear Regression
  https://towardsdatascience.com/assumptions-in-linear-regression-528bb7b0495d

  4. No Multicollinearity 
    — Multicollinearity is defined as the degree of inter-correlations among the independent variables used in the model. 
    - It is assumed that the independent feature variables are not at all or very less correlated among each other, which makes them independent. 
    - So in practical implementation, the correlation between two independent features must not be greater than 30% as it weakens 
      the statistical power of the model built. For identification of highly correlated features, pair plots (scatter plot) and heatmaps 
      (correlation matrix) can be used.

-------------------------------------------------------------------------------------
Question 65:
A credit card company wants to build a credit scoring model to help predict whether a new credit card applicant will default on a 
credit card payment. The company has collected data from a large number of sources with thousands of raw attributes. Early experiments 
to train a classification model revealed that many attributes are highly correlated, the large number of features slows down the training 
speed significantly, and that there are some overfitting issues. The Data Scientist on this project would like to speed up the model training 
time without losing a lot of information from the original dataset. Which feature engineering technique should the Data Scientist use to meet 
the objectives?

    Run self-correlation on all features and remove highly correlated features.
    Normalize all numerical values to be between 0 and 1.
  x Use an autoencoder or Principal Component Analysis (PCA) to replace original features with new features.
    Cluster raw data using K-means and use sample data from each cluster to build a new dataset.

-------------------

Comparing GAN, Autoencoder and VAE in 2024
  https://ubiai.tools/comparing-gan-autoencoders-and-vaes/

Autoencoders (AEs)
  - Autoencoders are neural networks that lower the dimensionality of input data by encoding it into a lower-dimensional representation, 
    which is then used to reconstruct the output.
  - Autoencoders consist of two vital components: an encoder which compresses the input data and a decoder which reconstruct the original 
    data from this reduced representation.

                  original input -----> Encoder -----> Compressed       ---------> Decoder ------> Reconstructed input
                                                       Representation

  - While this process may appear as a simple round trip, it serves as the gateway to unlocking the concept of feature learning because 
    this technique have a remarkable capability to remove noise and enhance image quality, detecting anomalies and reducing the size of 
    data while retaining essential information.


Generative Adversarial Networks (GAN)

  - A GAN is a generative model that is trained using two neural network models by treating the unsupervised problem as supervised and 
    using both a generative and a discriminative model. 
  - The generator’s role is to create synthetic outputs that closely resemble authentic data, often to the point of being indistinguishable 
    from real data.

  - The discriminator’s purpose is to determine which of the presented outputs are the result of artificial generation. 
  - It is a binary classifier that assigns a probability score to each data sample.

                    Real Image   -----> Sample   ----->                         Real 
                                                       Discriminator  ------->   or
                   Random Input  -----> Genertor ----->                         Fake

                                       GAN Architecture Diagram


-------------------------------------------------------------------------------------
Question 66:
A Data Scientist is training a multilayer perception (MLP) on a dataset with multiple classes. The target class of interest 
is unique compared to the other classes within the dataset, but it does not achieve and acceptable recall metric. The Data 
Scientist has already tried varying the number and size of the MLP's hidden layers, which has not significantly improved the 
results. A solution to improve recall must be implemented as quickly as possible. Which techniques should be used to meet 
these requirements?

    Gather more data using Amazon Mechanical Turk and then retrain.
    Train an anomaly detection model instead of an MLP.
    Train an XGBoost model instead of an MLP.
  x Add class weights to the MLP's loss function and then retrain.

-------------------

Multi-Layer Perceptron Learning in Tensorflow
  https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/

  What is a Multilayer Perceptron (MLP)?
   - A MLP consists of fully connected dense layers that transform input data from one dimension to another. 
   - It is called “multi-layer” because it contains an input layer, one or more hidden layers, and an output layer. 
   - The purpose of an MLP is to model complex relationships between inputs and outputs, making it a powerful tool for 
     various machine learning tasks.

  The key components of Multi-Layer Perceptron includes:
    Input Layer: 
      - Each neuron (or node) in this layer corresponds to an input feature. For instance, if you have three input features, 
        the input layer will have three neurons.
    Hidden Layers: 
      - An MLP can have any number of hidden layers, with each layer containing any number of nodes. 
      - These layers process the information received from the input layer.
    Output Layer: 
      - The output layer generates the final prediction or result. 
      - If there are multiple outputs, the output layer will have a corresponding number of neurons.

    Fully Connected 
      - every node in one layer connects to every node in the next layer. 
      - As the data moves through the network, each layer transforms it until the final output is generated in the output layer.

  What is the function of MLP?
    - The function of a Multi-Layer Perceptron (MLP) is to map input data to an output by learning patterns and 
      relationships through multiple layers of interconnected neurons. 
  What are the applications of MLP?
    - MLPs are used in applications like image recognition, natural language processing, speech recognition, 
      financial forecasting, and medical diagnosis.

   MLP vs CNN
     - MLPs are good for simple image classification, while CNNs are better for more complex image classification.

-------------------

Use weighted loss function to solve imbalanced data classification problems
  https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75

  - Imbalanced datasets are a common problem in classification tasks, where number of instances in one class is significantly 
    smaller than number of instances in another class. This will lead to biased models that perform poorly on minority class.

  - A weighted loss function is a modification of standard loss function used in training a model. 
  - The weights are used to assign a higher penalty to mis classifications of minority class. 
  - The idea is to make model more sensitive to minority class by increasing cost of mis classification of that class.

  - The most common way to implement a weighted loss function is to assign higher weight to minority class and lower weight 
    to majority class. 
  - The weights can be inversely proportional to frequency of classes, so that minority class gets higher weight and majority 
    class gets lower weight.

-------------------------------------------------------------------------------------
Question 68:
A Machine Learning Specialist needs to move and transform data in preparation for training. Some of the data needs to be processed in near-real time, and other data can be moved hourly. There are existing Amazon EMR MapReduce jobs to clean and feature engineering to perform on the data. Which of the following services can feed data to the MapReduce jobs? (Choose two.)

    AWS DMS.
  x Amazon Kinesis.
  x AWS Data Pipeline.
    Amazon Athena.
    Amazon ES.

-------------------
Process Streaming Data with Kinesis and Elastic MapReduce
  https://aws.amazon.com/blogs/aws/process-streaming-data-with-kinesis-and-elastic-mapreduce/

  New EMR Connector to Kinesis
    - Today we are adding an 'Elastic MapReduce Connector' to 'Kinesis.' 
    - With this connector, you can analyze streaming data using familiar Hadoop tools such as Hive, Pig, Cascading, and Hadoop Streaming. 
    - If you build the analytical portion of your streaming data application around the combination of Kinesis and Amazon Elastic MapReduce, 
      you will benefit from the fully managed nature of both services. 
    - You won’t have to worry about building deploying, or maintaining the infrastructure needed to do real-time processing at world-scale. 
    - This connector is available in version 3.0.4 of the Elastic MapReduce AMI.
-------------------
  AWS Data Pipeline is no longer available to new customers. Existing customers of AWS Data Pipeline can continue to use the service as normal

  Process Data Using Amazon EMR with Hadoop Streaming
    - You can use AWS Data Pipeline to manage your Amazon EMR clusters. 
    - With AWS Data Pipeline you can specify preconditions that must be met before the cluster is launched (for example, ensuring that 
      today's data been uploaded to Amazon S3), a schedule for repeatedly running the cluster, and the cluster configuration to use. 

-------------------------------------------------------------------------------------
Question 72:
A Machine Learning Specialist wants to determine the appropriate SageMakerVariantInvocationsPerInstance setting for 
an endpoint automatic scaling configuration. The Specialist has performed a load test on a single instance and determined 
that peak requests per second (RPS) without service degradation is about 20 RPS. As this is the first deployment, the Specialist 
intends to set the invocation safety factor to 0.5. Based on the stated parameters and given that the invocations per instance 
setting is measured on a per-minute basis, what should the Specialist set as the SageMakerVariantInvocationsPerInstance setting?

    10.
    30.
  x 600.
    2,400.


SageMakerVariantInvocationsPerInstance = (MAX_RPS * SAFETY_FACTOR) * 60 = 20 * 0.5 * 60 = 600

-------------------
After you find the performance characteristics of the variant, you can determine the maximum RPS we should allow to be sent to an instance. 
The threshold used for scaling must be less than this maximum value. Use the following equation in combination with load testing to determine 
the correct value for the SageMakerVariantInvocationsPerInstance target metric in your scaling configuration.

SageMakerVariantInvocationsPerInstance = (MAX_RPS * SAFETY_FACTOR) * 60

Where MAX_RPS is the maximum RPS that you determined previously, and SAFETY_FACTOR is the safety factor that you chose to ensure that your 
clients don't exceed the maximum RPS. Multiply by 60 to convert from RPS to invocations-per-minute to match the per-minute CloudWatch metric 
that SageMaker AI uses to implement auto scaling (you don't need to do this if you measured requests-per-minute instead of requests-per-second).

-------------------

 https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints-autoscaling.html
 SageMakerVariantInvocationsPerInstance predefined metric. 
   - SageMakerVariantInvocationsPerInstance is the average number of times per minute that each instance for 
     a variant is invoked. We strongly recommend using this metric.

  https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-add-code-define.html
  Specify a predefined metric (CloudWatch metric: InvocationsPerInstance)

   - The following is an example target tracking policy configuration for a variant that keeps the average 
     invocations per instance at 70. 
   - The policy configuration provides a scale-in cooldown period of 10 minutes (600 seconds) and a 
     scale-out cooldown period of 5 minutes (300 seconds). Save this configuration in a file named config.json.

    {
        "TargetValue": 70.0,
        "PredefinedMetricSpecification":
        {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 600,
        "ScaleOutCooldown": 300
    }

-------------------------------------------------------------------------------------
Question 74:
A Data Scientist is building a model to predict customer churn using a dataset of 100 continuous numerical features. The 
Marketing team has not provided any insight about which features are relevant for churn prediction. The Marketing team 
wants to interpret the model and see the direct impact of relevant features on the model outcome. While training a Logistic 
Regression model, the Data Scientist observes that there is a wide gap between the training and validation set accuracy. 
Which methods can the Data Scientist use to improve the model performance and satisfy the Marketing team's needs? (Choose two.)

  x Add L1 regularization to the classifier.
    Add features to the dataset.
  x Perform recursive feature elimination.
    Perform t-distributed stochastic neighbor embedding (t-SNE).
    Perform linear discriminant analysis.

-------------------

 note: (from Question 51's answer info)  WhizLabs Quiz 3 
 Linear Discriminant Analysis (LDA) is used to reduce dimensionality in multi-class classification problems that predict a categorical target. 
 We are trying to solve a continuous target, match point difference or spread.

-------------------

Feature Selection with “Recursive Feature Elimination” (RFE) for Parisian Bike Count Data
https://medium.com/@hsu.lihsiang.esth/feature-selection-with-recursive-feature-elimination-rfe-for-parisian-bike-count-data-23f0ce9db691

What is “Recursive Feature Elimination” (RFE)?

  - In broader terms, Recursive Feature Elimination is an iterative feature selection method that works by recursively removing 
    features from the dataset and evaluating the performance of a machine learning model at each step. 
  - It starts with all features and ranks them based on their importance or relevance to the target variable, and then removes the 
    least important feature(s) and repeats the process until the desired model performance is reached.

  - In narrower terms, Recursive Feature Elimination (RFE) specifically refers to a function within the Scikit-learn library’s 
    feature selection module. 
  - This function operationalizes the RFE approach by automating the iterative feature selection process. 
  - It streamlines the task by internally managing the iterative steps and offers the possibility for specifying parameters such 
    as the estimator (i.e., the machine learning model) and the desired number of features to select. 
  - By leveraging the RFE function, analysts can automate feature selection, saving time and ensuring consistency and reproducibility 
    in their analyses.

  - It allows the use of various machine learning models, such as linear regression, SVM, Extra-Trees, or Random Forest, leveraging 
    their feature importance scores for evaluation. 
  - The iterative process continues until the specified stopping criteria are met. 
  - Additionally, RFECV is combined with cross-validation techniques to further enhance the robustness of feature selection.


https://www.analyticsvidhya.com/blog/2023/05/recursive-feature-elimination/
Recursive Feature Elimination algorithm works in the following steps:

 1. Rank the importance of all features using the chosen RFE machine learning algorithm.
 2. Eliminate the least important feature.
 3. Build a model using the remaining features.
 4. Repeat steps 1-3 until the desired number of features is reached.

Few Other Feature Selection Methods:
  Filtering Method
    - A common method of Recursive feature selection is the filtering method. 
    - This method evaluates each feature individually and selects the most meaningful features based on statistical measures 
      such as correlation and mutual information. 
    - Filtering techniques are quick and easy to implement but may not consider interactions between features and may not be 
      effective with high-dimensional datasets.

  Wrapper Method
    - Another common method is a wrapper method that uses a learning algorithm that evaluates the usefulness of each subset of functions. 
    - Wrapper methods are more computationally expensive than filter methods but can consider the interactions between features and may 
      be more effective in high-dimensional datasets. 
    - However, they are more prone to overfitting and may be sensitive to the choice of learning algorithm.

  Principal Component Analysis (PCA)
    - Another method often compared to Recursive Feature Elimination is principal component analysis (PCA). 
    - It transforms features into a low-dimensional space that captures the most important information. 
    - PCA is an effective way to reduce the dimensionality of datasets and remove redundant features. 
    - Still, it may not preserve the interpretability of the original features and may not be suitable for non-linear relationships 
      between features. There is nature.

  - Compared to filter and wrapper methods, RFE has the advantage of considering both features’ relevance, redundancy, and interactions. 
  - By recursively removing the least important features, RFE can effectively reduce the dataset’s dimensionality while preserving 
    the most informative features. However, RFE can be computationally intensive and unsuitable for large datasets.

-------------------
AI generated:
 - Linear discriminant analysis (LDA) is a supervised machine learning technique that uses dimensionality reduction to classify data. 
 - It's also known as discriminant function analysis (DFA) or normal discriminant analysis (NDA). 

-------------------------------------------------------------------------------------
Question 77:

A Machine Learning Specialist is developing a daily ETL workflow containing multiple ETL jobs. The workflow consists of the following 
processes: Start the workflow as soon as data is uploaded to Amazon S3. When all the datasets are available in Amazon S3, start an 
ETL job to join the uploaded datasets with multiple terabyte-sized datasets already stored in Amazon S3. Store the results of joining 
datasets in Amazon S3. If one of the jobs fails, send a notification to the Administrator. Which configuration will meet these requirements?

  x Use AWS Lambda to trigger an AWS Step Functions workflow to wait for dataset uploads to complete in Amazon S3. Use AWS Glue to 
      join the datasets. Use an Amazon CloudWatch alarm to send an SNS notification to theAdministrator in the case of a failure.
    Develop the ETL workflow using AWS Lambda to start an Amazon SageMaker notebook instance. Use a lifecycle configuration script 
      to join the datasets and persist the results in Amazon S3. Use an Amazon CloudWatch alarm to sendan SNS notification to the 
      Administrator in the case of a failure.
    Develop the ETL workflow using AWS Batch to trigger the start of ETL jobs when data is uploaded to Amazon S3. Use AWS Glue to 
      join the datasets in Amazon S3. Use an Amazon CloudWatch alarm to send an SNS notification to theAdministrator in the case of a failure.
    Use AWS Lambda to chain other Lambda functions to read and join the datasets in Amazon S3 as soon as the data is uploaded to Amazon S3. 
      Use an Amazon CloudWatch alarm to send an SNS notification to the Administrator in the case of a failure.

-------------------
AWS Certified Machine Learning - Specialty (MLS-C01)
9.4 Using AWS Step Functions to Categorize Uploaded Data

About this lab

  AWS Step Functions is a powerful service that lets you build and orchestrate different AWS services to construct 
  state machines. This can be done with almost any AWS API action, and with little-to-no code.

  In this lab, we're going to build a Step Functions state machine to process an MP3 call recording, determine 
  the sentiment of the conversation, and take a different action depending on the analysis.

Learning objectives
  - Create the Step Function State Machine
  - Implement Amazon Transcribe to the State Machine
  - Implement Amazon Comprehend to the State Machine
  - Develop a Choice state based on Sentiment


Flow
                      |------------------------------------------------|   Positive
                      | Step Function                                  |      |---> SQS
                      |                                                |      |
    Audo File (MP3)   | ---> Transcribe --> Comprehend --> Sentiment ---------|
      S3              |                                      ??        |      |
                      |                                                |      |---> Lambda
                      |------------------------------------------------|    Negative

-------------------
A Cloud Guru AWS Certified Solutions Architect - Associate (SAA-C02)
13.9 Coordinating Distributed Apps with AWS Step Functions

    AWS Step Functions
      What is AWS Step Functions:
        - Serverless orchestration service meant for event-driven task executions combining Lambda 
          functions with different AWS services for business applications
        - comes with a graphical interface
      Execution Types:
        Standard: good for long-running, auditable executions
        Express workflows: good for high-event-rate executions
      Amazon States Language (ASL)
        - all state machines are written in the Amazon States Language format
        - proprietary format similar to JSON
      States
        - elements within your state machines
        - these are things and actions that happen with workflows
      Integrated AWS Services that work well with AWS Step Function
         Lambda, Batch, ECS/AWS Fargate, SNS, SQS, API Gateway, EventBridge, DynamoDB, etc.
      State Types:
        - there are currently 8 different state types: Pass, Task, Choice, Wait, Succeed, Fail, Parallel, and Map


-------------------------------------------------------------------------------------
Question 78:
An agency collects census information within a country to determine healthcare and social program needs by province and city. 
The census form collects responses for approximately 500 questions from each citizen. Which combination of algorithms would 
provide the appropriate insights? (Select TWO.)

    The factorization machines (FM) algorithm.
    The Latent Dirichlet Allocation (LDA) algorithm.
  x The Principal Component Analysis (PCA) algorithm.
  x The K-means algorithm.
    The Random Cut Forest (RCF) algorithm.

-------------------

 A Cloud Guru - AWS Certified Machine Learning - Specialty (MLS-C01)
    
    Clustering
      - unsupervised
      K-Means (kMeans)
        - an unsupervised learning algorithm for clustering
       - attempts to find discrete groupings within data, where members of a group are as similar as possible to one another and 
         as different as possible from members of other groups

    Recommendations
      Factorization Machines
        - is a general-purpose supervised learning algorithm used for both classification and regression tasks.
        - extension of a linear model designed to capture interactions between features within high dimensional sparse datasets 
          economically, such as click prediction and item recommendation.

    Topic Modeling
      Latent Dirichlet Allocation (LDA)
        - is an unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories.
        - used to discover a user-specified number of topics shared by documents within a text corpus.
      Neural Topic Mode (NTM)
        - is an unsupervised learning algorithm that is used to organize a corpus of documents into topics that contain word groupings 
          based on their statistical distribution
        - Topic modeling can be used to classify or summarize documents based on the topics detected or to retrieve information or 
          recommend content based on topic similarities.

    Feature Reduction
      PCA
        - is an unsupervised ML algorithm that attempts to reduce the dimensionality (number of features) within a dataset while still 
          retaining as much information as possible.
      Object2Vec
        - is a general-purpose neural embedding algorithm that is highly customizable
        - can learn low-dimensional dense embeddings of high-dimensional objects.

    Anomaly Detection
      Random Cut Forest
        - is an unsupervised algorithm for detecting anomalous data points within a data set.
      IP Insights
        - is an unsupervised learning algorithm that learns the usage patterns for IPv4 addresses.
        - designed to capture associations between IPv4 addresses and various entities, such as user IDs or account numbers

-------------------------------------------------------------------------------------
Question 79:

A large consumer goods manufacturer has the following products on sale: 34 different toothpaste variants. 48 different toothbrush 
variants. 43 different mouthwash variants. The entire sales history of all these products is available in Amazon S3. Currently, 
the company is using custom-built autoregressive integrated moving average (ARIMA) models to forecast demand for these products. 
The company wants to predict the demand for a new product that will soon be launched. Which solution should a Machine Learning 
Specialist apply?

    Train a custom ARIMA model to forecast demand for the new product.
  x Train an Amazon SageMaker DeepAR algorithm to forecast demand for the new product.
    Train an Amazon SageMaker K-means clustering algorithm to forecast demand for the new product.
    Train a custom XGBoost model to forecast demand for the new product.

-------------------
  Note: 'AR' in DeepAR stands for AutoRegressive
-------------------

Documentation -> Amazon SageMaker -> Developer Guide -> Use the SageMaker DeepAR forecasting algorithm
  https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html

  - The Amazon SageMaker DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar 
    (one-dimensional) time series using recurrent neural networks (RNN). 
  - Classical forecasting methods, such as autoregressive integrated moving average (ARIMA) or exponential smoothing (ETS), 
    fit a single model to each individual time series. They then use that model to extrapolate the time series into the future. 

  - We recommend training a DeepAR model on as many time series as are available. 
  - Although a DeepAR model trained on a single time series might work well, standard forecasting algorithms, such as ARIMA 
    or ETS, might provide more accurate results. 
  - The DeepAR algorithm starts to outperform the standard methods when your dataset contains hundreds of related time series. 
    Currently, DeepAR requires that the total number of observations available across all training time series is at least 300.

-------------------

Autoregressive Integrated Moving Average (ARIMA) Prediction Model
  https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp

 What Is an Autoregressive Integrated Moving Average (ARIMA)?
  - An autoregressive integrated moving average, or ARIMA, is a statistical analysis model that uses time series data to 
    either better understand the data set or to predict future trends. 
  - A statistical model is autoregressive if it predicts future values based on past values. 
  - For example, an ARIMA model might seek to predict a stock's future prices based on its past performance or forecast 
   a company's earnings based on past periods.

  Key Takeaways
    - Autoregressive integrated moving average (ARIMA) models predict future values based on past values.
    - ARIMA makes use of lagged moving averages to smooth time series data.
    - They are widely used in technical analysis to forecast future security prices.
    - Autoregressive models implicitly assume that the future will resemble the past.
    - Therefore, they can prove inaccurate under certain market conditions, such as financial crises or periods 
      of rapid technological change.


 An ARIMA model can be understood by outlining each of its components as follows:
    Autoregression (AR): 
      - refers to a model that shows a changing variable that regresses on its own lagged, or prior, values.
    Integrated (I): 
      - represents the differencing of raw observations to allow the time series to become stationary (i.e., data values 
        are replaced by the difference between the data values and the previous values).
    Moving average (MA):  
      - incorporates the dependency between an observation and a residual error from a moving average model applied 
      to lagged observations.

-------------------------------------------------------------------------------------
Question 80:

A Machine Learning Specialist uploads a dataset to an Amazon S3 bucket protected with server-side encryption using AWS KMS. How should the ML Specialist define the Amazon SageMaker notebook instance so it can read the same dataset from Amazon S3?

    Define security group(s) to allow all HTTP inbound/outbound traffic and assign those security group(s) to the Amazon SageMaker notebook instance.
    Configure the Amazon SageMaker notebook instance to have access to the VPC. Grant permission in the KMS key policy to the notebook's KMS role.
  x Assign an IAM role to the Amazon SageMaker notebook with S3 read access to the dataset. Grant permission in the KMS key policy to that role.
    Assign the same KMS key used to encrypt data in Amazon S3 to the Amazon SageMaker notebook instance.

-------------------
How do I resolve Amazon S3 AccessDenied errors in Amazon SageMaker training jobs?
  https://repost.aws/knowledge-center/sagemaker-s3-accessdenied-training

  Encrypted input bucket
    - If the data in the 'S3 bucket' is encrypted with AWS Key Management Service (AWS KMS), then check these permissions:
      - Be sure that the 'IAM policy' that's attached to the 'execution role' allows the 'kms:encrypt' and 'kms:decrypt' actions. 
    - Be sure that the 'AWS KMS key policy' grants access to the 'IAM role'. 
    - If you use an AWS KMS key for the machine learning (ML) storage volume in the resource configuration of your job, 
      then the IAM policy must allow kms:CreateGrant action. 

-------------------------------------------------------------------------------------
Question 85:

A Data Scientist wants to gain real-time insights into a data stream of GZIP files. Which solution would allow the use of SQL 
to query the stream with the LEAST latency?

  x Amazon Kinesis Data Analytics with an AWS Lambda function to transform the data.
    AWS Glue with a custom ETL script to transform the data.
    An Amazon Kinesis Client Library to transform the data and save it to an Amazon ES cluster.
    Amazon Kinesis Data Firehose to transform the data and put it into an Amazon S3 bucket.
  
-------------------

Amazon Kinesis Analytics can now pre-process data prior to running SQL queries
  https://aws.amazon.com/about-aws/whats-new/2017/10/amazon-kinesis-analytics-can-now-pre-process-data-prior-to-running-sql-queries/

  Posted On: Oct 5, 2017

  - You can now configure your Amazon Kinesis Analytics applications to transform data before it is processed by your SQL code. 
  - This new feature allows you to use AWS Lambda to convert formats, enrich data, filter data, and more. 
  - Once the data is transformed by your function, Kinesis Analytics sends the data to your application’s SQL code for real-time analytics.

  - To get started, simply select an AWS Lambda function from the Kinesis Analytics application source page in the AWS Management console. 
  - Your Kinesis Analytics application will automatically process your raw data records using the Lambda function, and send transformed 
   data to your SQL code for further processing.

-------------------------------------------------------------------------------------
Question 87:

A Machine Learning Specialist is assigned a TensorFlow project using Amazon SageMaker for training, and needs to continue working 
for an extended period with no Wi-Fi access. Which approach should the Specialist use to continue working?

    Install Python 3 and boto3 on their laptop and continue the code development using that environment.
  x Download the TensorFlow Docker container used in Amazon SageMaker from GitHub to their local environment, and use the Amazon 
      SageMaker Python SDK to test the code.
    Download TensorFlow from tensorflow.org to emulate the TensorFlow kernel in the SageMaker environment.
    Download the SageMaker notebook to their local environment, then install Jupyter Notebooks on their laptop and continue 
      the development in a local notebook.

-------------------

AWS Machine Learning Blog
Use the Amazon SageMaker local mode to train on your notebook instance
  https://aws.amazon.com/blogs/machine-learning/use-the-amazon-sagemaker-local-mode-to-train-on-your-notebook-instance/

  - Amazon SageMaker Python SDK supports local mode, which allows you to create estimators and deploy them to your local environment. 
  - This is a great way to test your deep learning scripts before running them in SageMaker’s managed training or hosting environments. 
  - Local Mode is supported for frameworks images (TensorFlow, MXNet, Chainer, PyTorch, and Scikit-Learn) and images you supply yourself.

  - The Amazon SageMaker deep learning containers allow you to write TensorFlow, PyTorch or MXNet scripts as you typically would. 
  - However, now you deploy them to pre-built containers in a managed, production-grade environment for both training and hosting. 
  - Previously, these containers were only available within these Amazon SageMaker-specific environments. 
  - They’ve recently been open sourced, which means you can pull the containers into your working environment and use custom code built 
    into the Amazon SageMaker Python SDK to test your algorithm locally, just by changing a single line of code. 
  - This means that you can iterate and test your work without having to wait for a new training or hosting cluster to be built each time. 
  - Iterating with a small sample of the dataset locally and then scaling to train on the full dataset in a distributed manner is 
    common in machine learning. 
  - Typically this would mean rewriting the entire process and hoping not to introduce any bugs. 
  - The Amazon SageMaker local mode allows you to switch seamlessly between local and distributed, managed training by simply changing 
    one line of code. Everything else works the same.

-------------------------------------------------------------------------------------
Question 90:

A Data Scientist is developing a machine learning model to classify whether a financial transaction is fraudulent. The labeled 
data available for training consists of 100,000 non-fraudulent observations and 1,000 fraudulent observations. The Data Scientist 
applies the XGBoost algorithm to the data, resulting in the following confusion matrix when the trained model is applied to a previously 
unseen validation dataset. The accuracy of the model is 99.1%, but the Data Scientist has been asked to reduce the number of false 
negatives. Which combination of steps should the Data Scientist take to reduce the number of false positive predictions by the model? 
(Choose two.)

                 Predicted
                   0      1
    Actual    0  99966    34
              1    877   123

Question 90

    Change the XGBoost eval_metric parameter to optimize based on rmse instead of error.
  x Increase the XGBoost scale_pos_weight parameter to adjust the balance of positive and negative weights.
    Increase the XGBoost max_depth parameter because the model is currently underfitting the data.
  x Change the XGBoost eval_metric parameter to optimize based on AUC instead of error.
    Decrease the XGBoost max_depth parameter because the model is currently overfitting the data.

  Note:

      scale_pos_weight = sum(negative cases) / sum(positive cases)   = 100000 / 1000 =  100

-------------------
XGBoost hyperparameters
https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html

   max_depth: 
      - Maximum depth of a tree. Increasing this value makes the model more complex and likely to be overfit. 
        0 indicates no limit. A limit is required when grow_policy=depth-wise.
      - Optional; Valid values: Integer. Range: [0,∞); Default value: 6`
   scale_pos_weight: 	
     - Controls the balance of positive and negative weights. It's useful for unbalanced classes. 
     - A typical value to consider: sum(negative cases) / sum(positive cases).
     - Optional; Valid values: float; Default value: 1
   eval_metric: 	
     - Evaluation metrics for validation data. A default metric is assigned according to the objective:
         - rmse: for regression
         - error: for classification
         - map: for ranking
     - For a list of valid inputs, see XGBoost Learning Task Parameters
       https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst#learning-task-parameters
         auc: Receiver Operating Characteristic Area under the Curve. Available for classification and learning-to-rank tasks.
            - When used with binary classification, the objective should be binary:logistic or similar functions that work on probability.
     - Optional; Valid values: String.; Default value: Default according to objective.

   objective 	
     - Specifies the learning task and the corresponding learning objective. 
     - Examples: reg:logistic, multi:softmax, reg:squarederror. For a full list of valid inputs, refer to XGBoost Learning Task Parameters
     - Optional;  Valid values: String;  Default value: "reg:squarederror"
     Note:
       multi:softmax: 
         - set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)

-------------------------------------------------------------------------------------
-------------------

   Redshift Spectrum vs Athena S3 queries:
     Redshift Spectrum S3 Queries
       - Query S3 Data
       - must have a Redshift cluster spun up.
       - Made for existing Redshift customers
     Athena S3 Queries
       - Query S3 Data
       - No need for Redshift cluster 
       - New customers quickly want to query S3 data
       - just need a table defined in your AWS Glue Ca
-------------------------------------------------------------------------------------
Question 94:
A company's Machine Learning Specialist needs to improve the training speed of a time-series forecasting model using TensorFlow. 
The training is currently implemented on a single-GPU machine and takes approximately 23 hours to complete. The training needs to 
be run daily. The model accuracy is acceptable, but the company anticipates a continuous increase in the size of the training data and 
a need to update the model on an hourly, rather than a daily, basis. The company also wants to minimize coding effort and infrastructure 
changes. What should the Machine Learning Specialist do to the training solution to allow it to scale for future demand?

    Do not change the TensorFlow code. Change the machine to one with a more powerful GPU to speed up the training.
  x Change the TensorFlow code to implement a Horovod distributed framework supported by Amazon SageMaker. Parallelize the training 
      to as many machines as needed to achieve the business goals.
    Switch to using a built-in AWS SageMaker DeepAR model. Parallelize the training to as many machines as needed to achieve the 
      business goals.
    Move the training to Amazon EMR and distribute the workload to as many machines as needed to achieve the business goals.

-------------------

Horovod
https://github.com/horovod/horovod

  - Horovod is a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. 
    The goal of Horovod is to make distributed deep learning fast and easy to use.

  - The primary motivation for this project is to make it easy to take a single-GPU training script and successfully scale it 
    to train across many GPUs in parallel.


Multi-GPU and distributed training using Horovod in Amazon SageMaker Pipe mode
  https://aws.amazon.com/blogs/machine-learning/multi-gpu-and-distributed-training-using-horovod-in-amazon-sagemaker-pipe-mode/

  - In this post, I explain how to run multi-GPU training on a single instance on Amazon SageMaker, and discuss efficient 
    multi-GPU and multi-node distributed training on Amazon SageMaker.

  Basics on Horovod
   - When you train a model with a large amount of data, you should distribute the training across multiple GPUs on either 
     a single instance or multiple instances. 
   - Deep learning frameworks provide their own methods to support multi-GPU training or distributed training. 
   - However, there is another way to accomplish this using distributed deep learning framework such as Horovod. 
   - Horovod is Uber’s open-source framework for distributed deep learning, and it’s available for use with most popular deep 
     learning toolkits like TensorFlow, Keras, PyTorch, and Apache MXNet. 
   - It uses the all-reduce algorithm for fast distributed training rather than using a parameter server approach, and includes 
     multiple optimization methods to make distributed training faster. 

-------------------------------------------------------------------------------------
Question 96:
An office security agency conducted a successful pilot using 100 cameras installed at key locations within the main office. Images from the cameras were uploaded to Amazon S3 and tagged using Amazon Rekognition, and the results were stored in Amazon ES. The agency is now looking to expand the pilot into a full production system using thousands of video cameras in its office locations globally. The goal is to identify activities performed by non-employees in real time Which solution should the agency consider?

  x Use a proxy server at each local office and for each camera, and stream the RTSP feed to a unique Amazon Kinesis Video Streams video stream. 
      On each stream, use Amazon Rekognition Video and create a stream processor to detect faces from a collection of known employees, and alert 
      when non-employees are detected.
    Use a proxy server at each local office and for each camera, and stream the RTSP feed to a unique Amazon Kinesis Video Streams video stream. 
      On each stream, use Amazon Rekognition Image to detect faces from a collection of known employees and alert when non-employees are detected.
    Install AWS DeepLens cameras and use the DeepLens_Kinesis_Video module to stream video to Amazon Kinesis Video Streams for each camera. 
      On each stream, use Amazon Rekognition Video and create a stream processor to detect faces from a collection on each stream, and alert 
      when non-employees are detected.
    Install AWS DeepLens cameras and use the DeepLens_Kinesis_Video module to stream video to Amazon Kinesis Video Streams for each camera. 
      On each stream, run an AWS Lambda function to capture image fragments and then call Amazon Rekognition Image to detect faces from a collection 
      of known employees, and alert when non-employees are detected.


-------------------

[Rekognition] Working with streaming video events
https://docs.aws.amazon.com/rekognition/latest/dg/streaming-video.html


  - You can use Amazon Rekognition Video to detect and recognize faces or detect objects in streaming video. 
  - Amazon Rekognition Video uses Amazon Kinesis Video Streams to receive and process a video stream. 
  - You create a 'stream processor' with parameters that show what you want the stream processor to detect from the video stream. 
  - Rekognition sends label detection results from streaming video events as Amazon SNS and Amazon S3 notifications. 
  - Rekognition outputs face search results to a Kinesis data stream.

   
      Data Producers:         Kinesis Streams    Rekognition              Detection
                              Video Streams      Video                    Results       
      --------------------    ---------------    --------------           ---------------

      Web Cameras               -------->        Processes Video            
      Security Cameras          -------->        to detect faces   --->   SNS   
      audio feeds               -------->        and objects              S3 [event] Notifications
      radar data                -------->        objects           --->
                                -------->                             
                                --------> 


-------------------------------------------------------------------------------------
Question 98

An online reseller has a large, multi-column dataset with one column missing 30% of its data. A Machine Learning Specialist believes 
that certain columns in the dataset could be used to reconstruct the missing data. Which reconstruction approach should the 
Specialist use to preserve the integrity of the dataset?

    Listwise deletion.
    Last observation carried forward.
  x Multiple imputation.
    Mean substitution.

-------------------

Multiple Imputation
  https://dept.stat.lsa.umich.edu/~jerrick/courses/stat701/notes/mi.html


  Methods to handle missingness

    If the data is MCAR:
      - Complete case analysis is valid.
      - Mulitple imputation or any other imputation method is valid.

    If the data is MAR:
      - Some complete cases analyses are valid under weaker assumptions than MCAR.
          E.g. linear regression is unbiased if missingness is independent of the response, conditional on the predictors.
      - Multiple imputation is valid (it is biased, but the bias is negligible).

    If the data is MNAR:
      -  You must model the missingness explicitly; jointly modeling the response and missingness.
      -  In some specific cases (e.g. survival analysis), MNAR data (e.g. censored data) is handled appropriately.
      -  Generally, we assume MAR whenever possible just to avoid this situation.
 
  Multiple Imputation
    - The general procedure for Multiple Imputation is straightforward
      1. Impute the missing values with values randomly drawn from some distributions to generate 'm' complete cases data sets.
      2. Perform the same analysis on each of the 'm' datasets
      3. Pool the results in some fashion.

-------------------------------------------------------------------------------------

Question 100

A company is setting up an Amazon SageMaker environment. The corporate data security policy does not allow communication over the 
internet. How can the company enable the Amazon SageMaker service without enabling direct internet access to Amazon SageMaker 
notebook instances?

    Create a NAT gateway within the corporate VPC.
    Route Amazon SageMaker traffic through an on-premises network.
  x Create Amazon SageMaker VPC interface endpoints within the corporate VPC.
    Create VPC peering with Amazon VPC hosting Amazon SageMaker.

-------------------

  Connect to SageMaker Within your VPC
    https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html

    - You can connect directly to the SageMaker API or to Amazon SageMaker Runtime through an interface endpoint in your 
      virtual private cloud (VPC) instead of connecting over the internet. When you use a VPC interface endpoint, communication 
      between your VPC and the SageMaker API or Runtime is conducted entirely and securely within an AWS network. 

    - The SageMaker API and SageMaker Runtime support Amazon Virtual Private Cloud (Amazon VPC) interface endpoints that are 
      powered by AWS PrivateLink. 
    - Each VPC endpoint is represented by one or more Elastic Network Interfaces (ENI) with private IP addresses in your VPC subnets


-------------------------------------------------------------------------------------
Question 103:
-------------------

  Convert input data format in Amazon Data Firehose
    https://docs.aws.amazon.com/firehose/latest/dev/record-format-conversion.html

    - Amazon Data Firehose can convert the format of your input data from JSON to Apache Parquet or Apache ORC before storing the data in Amazon S3. 
    - Parquet and ORC are columnar data formats that save space and enable faster queries compared to row-oriented formats like JSON. 
    - If you want to convert an input format other than JSON, such as comma-separated values (CSV) or structured text, you can use AWS Lambda 
      to transform it to JSON first. 

-------------------------------------------------------------------------------------
Question 105:
A Machine Learning Specialist is implementing a full Bayesian network on a dataset that describes public transit in New York City. 
One of the random variables is discrete, and represents the number of minutes New Yorkers wait for a bus given that the buses cycle 
every 10 minutes, with a mean of 3 minutes. Which prior probability distribution should the ML Specialist use for this variable?

  x Poisson distribution.
    Uniform distribution.
    Normal distribution.
    Binomial distribution.

-------------------

A Cloud Guru:
AWS Certified Machine Learning - Specialty (MLS-C01): Exploratory Data Analysis

4.1 Understanding Probability Distributions
  . . .
  Discrete Distributions:
    Bernoulli Distribution
      - an event with a single trial with exactly two possible outcomes
      - The graph of a Bernoulli distribution is a simple bar chart with two values.
        The first bar indicates the outcome 1 (value of P).  The second bar indicates the outcome 2 (value of 1 - P).
  
    Binomial distribution
      https://medium.com/swlh/binomial-vs-bernoulli-distribution-dd9197c418d
      - repetition of multiple Bernoulli events
      - If Bernoulli distribution is an event with a single trial with exactly two possible outcomes, then binomial 
        distribution is nothing but repetition of multiple Bernoulli events.
      - coin toss example: if you repeat the trial n number of times where each trial is independent, the probability 
        of heads or tails is same for all the trials.
      - A binomial distribution is better represented as a histogram.
  
    Poisson distribution
      - the probability that an event will occur within a specific time
      - The rate of occurrence is known, but the actual timing of the occurrence is unknown.
      -  example: Predicting a hospital receiving emergency call. They know on an average they receive two 
         calls per day, but they cannot predict the exact timings.
      - This distribution relies on one parameter, X, which is the mean number of events.

      -  Poisson distribution is used to estimate how many times an event is likely to occur within the given period of time


  Continuous Distributions:
    Normal distribution
      - measure and visualize symmetrically distributed data with no skew
      - example:students' scores: most of the students' scores might range between 70 and 90, which forms the cluster at the center.
        Some of the top and bottom performers contribute to the tail at both the ends.  Parting the students' score will result 
        in a bell-shaped curve representing the normal distribution of the scores.

    Log-Normal distribution
      - derived from a normally distributed data and represents its logarithmic values
      - often used in financial data to understand future stock prices based on past trends.
      - lognormally distributed data does not form a symmetric shape but rather slants or skews more towards the right.

    Exponential distribution
      - models the time elapsed between the occurences of two events
      - reusing hospital receiving an emergency call example: exponential distribution models a time interval between two calls.
      - Exponential distribution relies on one parameter, R, which is the rate of occurrence.

-------------------------------------------------------------------------------------
Question 107:
A company is observing low accuracy while training on the default built-in image classification algorithm in Amazon SageMaker. The Data Science team wants to use an Inception neural network architecture instead of a ResNet architecture. Which of the following will accomplish this? (Choose two.)

    Customize the built-in image classification algorithm to use Inception and use this for model training.
    Create a support case with the SageMaker team to change the default image classification algorithm to Inception.
  x Bundle a Docker container with TensorFlow Estimator loaded with an Inception network and use this for model training.
  x Use custom code in Amazon SageMaker with TensorFlow Estimator to load the model with an Inception network, and use this 
      for model training.
    Download and apt-get install the inception network code into an Amazon EC2 instance and use this instance as a Jupyter 
      notebook in Amazon SageMaker.

-------------------------------------------------------------------------------------
Question 111:
A Machine Learning Specialist working for an online fashion company wants to build a data ingestion solution for the company's 
Amazon S3-based data lake. The Specialist wants to create a set of ingestion mechanisms that will enable future capabilities 
comprised of: Real-time analytics. Interactive analytics of historical data. Clickstream analytics. Product recommendations. 
Which services should the Specialist use?

  x AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for real-time data insights; 
      Amazon Kinesis Data Firehose for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized 
      product recommendations.
    Amazon Athena as the data catalog: Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for near-real-time data insights; 
      Amazon Kinesis Data Firehose for clickstream analytics; AWS Glue to generate personalized product recommendations.
    AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; 
      Amazon Kinesis Data Firehose for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized 
      product recommendations.
    Amazon Athena as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; 
      Amazon DynamoDB streams for clickstream analytics; AWS Glue to generate personalized product recommendations.

-------------------

What is Amazon OpenSearch Service?
https://docs.aws.amazon.com/opensearch-service/latest/developerguide/what-is.html

  - OpenSearch is a fully open-source search and analytics engine for use cases such as log analytics, real-time 
    application monitoring, and clickstream analysis.

OpenSearch Service includes the following features:

  Scale

    Numerous configurations of CPU, memory, and storage capacity known as instance types, including cost-effective Graviton instances

    Supports up to 1002 data nodes

    Up to 25 PB of attached storage

    Cost-effective UltraWarm and cold storage for read-only data


-------------------


Perform Near Real-time Analytics on Streaming Data with Amazon Kinesis and Amazon Elasticsearch Service
https://aws.amazon.com/blogs/big-data/perform-near-real-time-analytics-on-streaming-data-with-amazon-kinesis-and-amazon-elasticsearch-service/

August 30, 2023: Amazon Kinesis Data Analytics has been renamed to Amazon Managed Service for Apache Flink
Note: ElasticSearch is now OpenSearch

In this post, we use Amazon Kinesis Data Streams to collect and store streaming data. We then use Amazon Kinesis Data Analytics 
to process and analyze the streaming data continuously. Specifically, we use the Kinesis Data Analytics built-in RANDOM_CUT_FOREST 
function, a machine learning algorithm, to detect anomalies in the streaming data. Finally, we use Amazon Kinesis Data Firehose to 
export the anomalies data to Amazon OpenSearch Service. We then build a simple dashboard in the open source tool Kibana to visualize 
the result.

Solution overview

  The following diagram depicts a high-level overview of this solution.

   Sensors --> Kinesis        --> Kinesis         ---> Kinesis     ---> ElasticSearch
               Data Streams       Data Analytics       Firehose   

Amazon Kinesis Data Analytics

  Kinesis Data Analytics provides an easy and familiar standard SQL language to analyze streaming data in real time. 
  One of its most powerful features is that there are no new languages, processing frameworks, or complex machine learning 
  algorithms that you need to learn.

Amazon Kinesis Data Firehose

  Kinesis Data Firehose is the easiest way to load streaming data into AWS. It can capture, transform, and load streaming data 
  into Amazon S3, Amazon Redshift, and Amazon OpenSearch Service.

Amazon OpenSearch Service

  Amazon OpenSearch Service is a fully managed service that makes it easy to deploy, operate, and scale Elasticsearch for log 
  analytics, full text search, application monitoring, and more.

Solution summary

  The following is a quick walkthrough of the solution that’s presented in the diagram:

    1. IoT sensors send streaming data into Kinesis Data Streams. In this post, you use a Python script to simulate an IoT 
       temperature sensor device that sends the streaming data.
    2. By using the built-in RANDOM_CUT_FOREST function in Kinesis Data Analytics, you can detect anomalies in real time with 
      the sensor data that is stored in Kinesis Data Streams. RANDOM_CUT_FOREST is also an appropriate algorithm for many other kinds 
      of anomaly-detection use cases—for example, the media sentiment example mentioned earlier in this post.
    3. The processed anomaly data is then loaded into the Kinesis Data Firehose delivery stream.
    4. By using the built-in integration that Kinesis Data Firehose has with Amazon OpenSearch Service, you can easily export 
       the processed anomaly data into the service and visualize it with Kibana.

-------------------------------------------------------------------------------------
Question  115:
An agricultural company is interested in using machine learning to detect specific types of weeds in a 100-acre grassland field. 
Currently, the company uses tractor-mounted cameras to capture multiple images of the field as 10 - 10 grids. The company also has 
a large training dataset that consists of annotated images of popular weed classes like broadleaf and non-broadleaf docks. The company 
wants to build a weed detection model that will detect specific types of weeds and the location of each type within the field. Once 
the model is ready, it will be hosted on Amazon SageMaker endpoints. The model will perform real-time inferencing using the images 
captured by the cameras. Which approach should a Machine Learning Specialist take to obtain accurate predictions?

    Prepare the images in RecordIO format and upload them to Amazon S3. Use Amazon SageMaker to train, test, and validate the model 
      using an image classification algorithm to categorize images into various weed classes.
    Prepare the images in Apache Parquet format and upload them to Amazon S3. Use Amazon SageMaker to train, test, and validate the 
      model using an object- detection single-shot multibox detector (SSD) algorithm.
  x Prepare the images in RecordIO format and upload them to Amazon S3. Use Amazon SageMaker to train, test, and validate the model 
      using an object-detection single-shot multibox detector (SSD) algorithm.
    Prepare the images in Apache Parquet format and upload them to Amazon S3. Use Amazon SageMaker to train, test, and validate the 
      model using an image classification algorithm to categorize images into various weed classes.

-------------------

Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning
  https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab


The Region-Convolutional Neural Network (R-CNN)

  - A few years ago, by exploiting some of the leaps made possible in computer vision via CNNs, researchers developed R-CNNs 
    to deal with the tasks of object detection, localization and classification. 
  - Broadly speaking, a R-CNN is a special type of CNN that is able to locate and detect objects in images: the output is 
    generally a set of bounding boxes that closely match each of the detected objects, as well as a class output for each 
    detected object.

  - Fortunately, in the last few years, new architectures were created to address the bottlenecks of R-CNN and its successors, 
    enabling real-time object detection. 
  - The most famous ones are YOLO (You Only Look Once) and SSD MultiBox (Single Shot Detector)

Single Shot MultiBox Detector (SSD)

 - The paper about SSD: Single Shot MultiBox Detector (by C. Szegedy et al.) was released at the end of November 2016 and 
   reached new records in terms of performance and precision for object detection tasks, scoring over 74% mAP (mean Average 
   Precision) at 59 frames per second on standard datasets such as PascalVOC and COCO. 

 - To better understand SSD, let’s start by explaining where the name of this architecture comes from:

    Single Shot: 
      - this means that the tasks of object localization and classification are done in a single forward pass of the network
    MultiBox: 
      - this is the name of a technique for bounding box regression developed by Szegedy et al. 
    Detector: 
      - The network is an object detector that also classifies those detected objects

-------------------
AI Overview:

  - The main difference between image classification and object detection is that image classification categorizes an entire 
    image, while object detection identifies and locates specific objects within an image: 

  Image classification
    - Assigns a label to an entire image based on the presence of specific objects. 
    - For example, an e-commerce site might use image classification to categorize products into categories like lighting. 

  Object detection
    - Identifies objects in an image and marks their location with bounding boxes. 
    - For example, an autonomous driving system might use object detection to identify pedestrians, vehicles, and traffic signs. 

  - Both image classification and object detection use machine learning or deep learning to extract features from images 
    and make predictions. Convolutional Neural Networks (CNNs) are a common tool used for both tasks. 


-------------------------------------------------------------------------------------
Question 111:

A data scientist has explored and sanitized a dataset in preparation for the modeling phase of a supervised learning task. 
The statistical dispersion can vary widely between features, sometimes by several orders of magnitude. Before moving on to 
the modeling phase, the data scientist wants to ensure that the prediction performance on the production data is as accurate 
as possible. Which sequence of steps should the data scientist take to meet these requirements?


    Apply random sampling to the dataset. Then split the dataset into training, validation, and test sets.
  x Split the dataset into training, validation, and test sets. Then rescale the training set and apply the same scaling 
      to the validation and test sets.
    Rescale the dataset. Then split the dataset into training, validation, and test sets.
    Split the dataset into training, validation, and test sets. Then rescale the training set, the validation set, and 
      the test set independently.

-------------------

How to Prevent Data Leakage in Machine Learning?
  https://airbyte.com/data-engineering-resources/what-is-data-leakage

Here are some best practices that can significantly reduce the risk of data leakage and help you build more reliable and 
robust machine learning models:

  Proper Data Splitting: 
    - It is crucial to separate your data into distinct training and validation sets. Doing so ensures that no information 
      from the validation set leaks into the training set or vice versa. 
    - This separation ensures that the model is trained only on the training set, allowing it to learn patterns and relationships 
      in the data without any knowledge of the validation set. 
  . . .

  Feature Engineering: 
    - Feature engineering should be carried out exclusively using the training data. 
    - It is crucial to prevent utilizing any information from the validation or test sets to create new features, as 
      this can lead to data leakage.

  Preprocessing: 
    - Avoid preprocessing the data based on the entire dataset. 
    - Scaling, normalization, imputation, or any other data preprocessing steps should be performed solely on the training set.
  . . .

-------------------------------------------------------------------------------------
Question 116:
A manufacturer is operating a large number of factories with a complex supply chain relationship where unexpected downtime of a 
machine can cause production to stop at several factories. A data scientist wants to analyze sensor data from the factories to 
identify equipment in need of preemptive maintenance and then dispatch a service team to prevent unplanned downtime. The sensor 
readings from a single machine can include up to 200 data points including temperatures, voltages, vibrations, RPMs, and pressure 
readings. To collect this sensor data, the manufacturer deployed Wi-Fi and LANs across the factories. Even though many factory 
locations do not have reliable or high- speed internet connectivity, the manufacturer would like to maintain near-real-time 
inference capabilities. Which deployment architecture for the model will address these business requirements?

    Deploy the model in Amazon SageMaker. Run sensor data through this model to predict which machines need maintenance.
  x Deploy the model on AWS IoT Greengrass in each factory. Run sensor data through this model to infer which machines 
      need maintenance.
    Deploy the model to an Amazon SageMaker batch transformation job. Generate inferences in a daily batch report to identify 
      machines that need maintenance.
    Deploy the model in Amazon SageMaker and use an IoT rule to write data to an Amazon DynamoDB table. Consume a DynamoDB 
      stream from the table with an AWS Lambda function to invoke the endpoint.

-------------------

How AWS IoT Greengrass works
  https://docs.aws.amazon.com/greengrass/v2/developerguide/how-it-works.html

  - The AWS IoT Greengrass client software, also called AWS IoT Greengrass Core software, runs on Windows and Linux-based 
    distributions, such as Ubuntu or Raspberry Pi OS, for devices with ARM or x86 architectures. 
  - With AWS IoT Greengrass, you can program devices to act locally on the data they generate, run predictions based on 
    machine learning models, and filter and aggregate device data. 
  - AWS IoT Greengrass enables local execution of AWS Lambda functions, Docker containers, native OS processes, or custom 
    runtimes of your choice.

  - AWS IoT Greengrass provides pre-built software modules called components that let you easily extend edge device functionality. 
  - AWS IoT Greengrass components enable you to connect to AWS services and third-party applications at the edge. 
  - After you develop your IoT applications, AWS IoT Greengrass enables you to remotely deploy, configure, and manage those 
    applications on your fleet of devices in the field.

  - The following example shows how an AWS IoT Greengrass device interacts with the AWS IoT Greengrass cloud service and 
    other AWS services in the AWS Cloud.

        IoT thing group                                                                |----------------------|
          - organize from 1 to millions of Greengrass core devices     <---Deploy----- |  AWS IoT Greengrass  |
                                                                             |         |    cloud service     |
         |----------------------------------------------------------|        |         | -------------------  |
         |AWS IO Greengrass core device                             |        |         |    AWS IoT Core      |
         |                                                          |  <-----|         |                      |
         |        Lambda function         Bring your own runtime    |                  |    AWS IoT Analytics |
         |                                                          |                  |                      |
         |        Docker container        ... and more              | <-- Data   ----> |    SageMaker         |
         |     -------------------------------------------------    |    Exchange      |                      |
         |        Greengrass components:                            |                  |    Cloudwatch        |
         |              AWS IoT Greengrass client software          |                  |                      |
         |                                                          |                  |    S3                |
         |              operating system                            |                  |     ... and more     |
         |----------------------------------------------------------|                  |----------------------|


-------------------------------------------------------------------------------------
Question 117:
A Machine Learning Specialist is designing a scalable data storage solution for Amazon SageMaker. There is an existing 
TensorFlow-based model implemented as a train.py script that relies on static training data that is currently stored as TFRecords. 
Which method of providing training data to Amazon SageMaker would meet the business requirements with the LEAST development overhead?

    Use Amazon SageMaker script mode and use train.py unchanged. Point the Amazon SageMaker training invocation to the local 
      path of the data without reformatting the training data.
  x Use Amazon SageMaker script mode and use train.py unchanged. Put the TFRecord data into an Amazon S3 bucket. Point the 
      Amazon SageMaker training invocation to the S3 bucket without reformatting the training data.
    Rewrite the train.py script to add a section that converts TFRecords to protobuf and ingests the protobuf data instead 
      of TFRecords.
    Prepare the data in the format accepted by Amazon SageMaker. Use AWS Glue or AWS Lambda to reformat and store the data in 
      an Amazon S3 bucket.

-------------------
Use TensorFlow with the SageMaker Python SDK
  https://sagemaker.readthedocs.io/en/v2.66.1/frameworks/tensorflow/using_tf.html

  Training with Pipe Mode using PipeModeDataset

    - Amazon SageMaker allows users to create training jobs using Pipe input mode.
    - With Pipe input mode, your dataset is streamed directly to your training instances instead of being downloaded first. 
    - This means that your training jobs start sooner, finish quicker, and need less disk space.

    - SageMaker TensorFlow provides an implementation of 'tf.data.Dataset' that makes it easy to take advantage of Pipe 
      input mode in SageMaker. 
    - You can replace your 'tf.data.Dataset' with a 'sagemaker_tensorflow.PipeModeDataset' to read 'TFRecords' as they are 
      streamed to your training instances.

-------------------------------------------------------------------------------------

Question 119:
A retail company is using Amazon Personalize to provide personalized product recommendations for its customers during a marketing 
campaign. The company sees a significant increase in sales of recommended items to existing customers immediately after deploying 
a new solution version, but these sales decrease a short time after deployment. Only historical data from before the marketing 
campaign is available for training. How should a data scientist adjust the solution?

  x Use the event tracker in Amazon Personalize to include real-time user interactions.
    Add user metadata and use the HRNN-Metadata recipe in Amazon Personalize.
    Implement a new solution using the built-in factorization machines (FM) algorithm in Amazon SageMaker.
    Add event type and event value fields to the interactions dataset in Amazon Personalize.

-------------------
https://docs.aws.amazon.com/personalize/latest/dg/recording-events.html

Amazon Personalize -> Developer Guide -> Recording real-time events to influence recommendations

  - An event is an interaction between a user and your catalog. 
  - It can be an interaction with an item, such as a user purchasing an item or watching a video, or it can be taking an action, 
    such as applying for a credit card or enrolling in a membership program.

  - Amazon Personalize can make recommendations based on real-time event data only, historical event data only, or a mixture of both. 
  - Record real-time events as your customers interact with recommendations. This builds out your interactions data and keeps your 
    data fresh. 
  - And it tells Amazon Personalize about the current interests of your user, which can improve recommendation relevance. 

  - If your domain use case or custom recipe supports real-time personalization, Amazon Personalize uses events in real time to 
    update and adapt recommendations according to a user's evolving interest.

-------------------------------------------------------------------------------------
Question 120:
A machine learning (ML) specialist wants to secure calls to the Amazon SageMaker Service API. The specialist has configured 
Amazon VPC with a VPC interface endpoint for the Amazon SageMaker Service API and is attempting to secure traffic from specific 
sets of instances and IAM users. The VPC is configured with a single public subnet. Which combination of steps should the ML 
specialist take to secure the traffic? (Choose two.)

  x Add a VPC endpoint policy to allow access to the IAM users.
    Modify the users' IAM policy to allow access to Amazon SageMaker Service API calls only.
  x Modify the security group on the endpoint network interface to restrict access to the instances.
    Modify the ACL on the endpoint network interface to restrict access to the instances.
    Add a SageMaker Runtime VPC endpoint interface to the VPC.

-------------------

AWS Documentation -> Amazon VPC -> AWS PrivateLink -> Control access to VPC endpoints using endpoint policies
  https://docs.aws.amazon.com/vpc/latest/privatelink/vpc-endpoints-access.html

  - An endpoint policy is a resource-based policy that you attach to a VPC endpoint to control which AWS principals can use 
      the endpoint to access an AWS service.

  - An endpoint policy does not override or replace identity-based policies or resource-based policies. 
  - For example, if you're using an interface endpoint to connect to Amazon S3, you can also use Amazon S3 bucket policies to 
    control access to buckets from specific endpoints or specific VPCs.

-------------------------------------------------------------------------------------
Question 121:
An e commerce company wants to launch a new cloud-based product recommendation feature for its web application. Due to data 
localization regulations, any sensitive data must not leave its on-premises data center, and the product recommendation model 
must be trained and tested using nonsensitive data only. Data transfer to the cloud must use IPsec. The web application is 
hosted on premises with a PostgreSQL database that contains all the data. The company wants the data to be uploaded securely 
to Amazon S3 each day for model retraining. How should a machine learning specialist meet these requirements?

  x Create an AWS Glue job to connect to the PostgreSQL DB instance. Ingest tables without sensitive data through an 
      AWS Site-to-Site VPN connection directly into Amazon S3.
    Create an AWS Glue job to connect to the PostgreSQL DB instance. Ingest all data through an AWS Site-to-Site VPN 
      connection into Amazon S3 while removing sensitive data using a PySpark job.
    Use AWS Database Migration Service (AWS DMS) with table mapping to select PostgreSQL tables with no sensitive data 
      through an SSL connection. Replicate data directly into Amazon S3.
    Use PostgreSQL logical replication to replicate all data to PostgreSQL in Amazon EC2 through AWS Direct Connect with 
      a VPN connection. Use AWS Glue to move data from Amazon EC2 to Amazon S3.

-------------------
AWS -> Documentation -> AWS Glue -> User Guide -> AWS Glue connection properties
  https://docs.aws.amazon.com/glue/latest/dg/connection-properties.html

AWS Glue JDBC connection properties

  - AWS Glue can connect to the following data stores through a JDBC connection:
      Amazon Redshift
      Amazon Aurora
      Microsoft SQL Server
      MySQL
      Oracle
      PostgreSQL
      Snowflake, when using AWS Glue crawlers.
      Aurora (supported if the native JDBC driver is being used. Not all driver features can be leveraged)
      Amazon RDS for MariaDB

-------------------------------------------------------------------------------------
Question 122:
A logistics company needs a forecast model to predict next month's inventory requirements for a single item in 10 warehouses. 
A machine learning specialist usesAmazon Forecast to develop a forecast model from 3 years of monthly data. There is no missing 
data. The specialist selects the DeepAR+ algorithm to train a predictor. The predictor means absolute percentage error (MAPE) is 
much larger than the MAPE produced by the current human forecasters. Which changes to the CreatePredictor API call could improve 
the MAPE? (Choose two.)

  x Set PerformAutoML to true.
    Set ForecastHorizon to 4.
    Set ForecastFrequency to W for weekly.
  x Set PerformHPO to true.
    Set FeaturizationMethodName to filling.

-------------------
  https://docs.aws.amazon.com/forecast/latest/dg/API_CreatePredictor.html
  Amazon Forecast is no longer available to new customers. Existing customers of Amazon Forecast can continue to use the 
  service as normal

  After careful consideration, we have made the decision to close new customer access to Amazon Forecast, effective July 29, 2024.

   AutoML
    - If you want Amazon Forecast to evaluate each algorithm and choose the one that minimizes the objective function, set 
      PerformAutoML to true. T
   PerformHPO
    - Whether to perform hyperparameter optimization (HPO). 
    - HPO finds optimal hyperparameter values for your training data. 
    - The process of performing HPO is known as running a hyperparameter tuning job.
-------------------------------------------------------------------------------------
Question 123:
A data scientist wants to use Amazon Forecast to build a forecasting model for inventory demand for a retail company. The 
company has provided a dataset of historic inventory demand for its products as a .csv file stored in an Amazon S3 bucket. 
The table below shows a sample of the dataset. How should the data scientist transform the data?

    timestamp     item_id       demand       Category      lead_time
    ------------  -----------   --------     -----------   ----------
    2019-12-14    uni_000736    120          hardware       90
    2020-01-31    uni_003429     98          hardware       30
    2020-03-04    uni_000211    234          accessories    10

  x Use ETL jobs in AWS Glue to separate the dataset into a target time series dataset and an item metadata dataset. 
      Upload both datasets as .csv files to Amazon S3.
    Use a Jupyter notebook in Amazon SageMaker to separate the dataset into a related time series dataset and an item 
      metadata dataset. Upload both datasets as tables in Amazon Aurora.
    Use AWS Batch jobs to separate the dataset into a target time series dataset, a related time series dataset, and an 
      item metadata dataset. Upload them directly to Forecast from a local machine.
    Use a Jupyter notebook in Amazon SageMaker to transform the data into the optimized protobuf recordIO format. Upload 
      the dataset in this format to Amazon S3.

-------------------
Prepare and clean your data for Amazon Forecast
  https://aws.amazon.com/blogs/machine-learning/tailor-and-prepare-your-data-for-amazon-forecast/


  Factors affecting forecast accuracy
    - Amazon Forecast uses your data to train a private, custom model tailored to your use case. ML models are only as good 
      as the data put into them, and it’s important to understand what the model needs. 
    - Amazon Forecast can accept three types of datasets: target time series, related time series, and item metadata. 
    - Amongst those, target time series is the only mandatory dataset. 
    - This historical data provides the majority of the model’s accuracy.

  Target time series data
    - Target time series data defines the historical demand for the resources you’re predicting. 
    - The target time series dataset is mandatory. It contains three required fields:

      item_id 
        – Describes a unique identifier for the item or category you want to predict. 
        - This field may be named differently depending on your dataset domain (for example, in the workforce domain 
          this is workforce_type, which helps distinguish different groups of your labor force).
      timestamp 
        – Describes the date and time at which the observation was recorded.
      demand 
        – Describes the amount of the item, specified by item_id, that was consumed at the timestamp specified. 
        - For example, this could be the number of pink shoes sold on a certain day.

  - You can also add additional fields in your input data

  Related time series data
   - In addition to historical sales data, other data may be known per item at exactly the same time as every sale. 
   - This data is called related time series data. 
   - Related data can give more clues to what future predictions could look like. 
   - The best related data is also known in the future. 
   - Examples of related data include prices, promotions, economic indicators, holidays, and weather. 
   - Although related time series data is optional, including additional information can help increase accuracy by 
     providing context of various conditions that may have affected demand.

  Item metadata
    - Providing item metadata to Amazon Forecast is optional, but can help refine forecasts by adding contextual information 
      about items that appear in your target time series data.     
    - Item metadata is static information that doesn’t change with time, describing features about items such as the color 
      and size of a product being sold. Amazon Forecast uses this data to create predictions based on similarities between products.

    - To use item metadata, you upload a separate file to Amazon Forecast. 
    - Each row in the CSV file you upload must contain the item ID, followed by the metadata features for that item. 
    - Each row can have a maximum of 10 fields, including the field that contained the item ID.

    - Item metadata is required when forecasting demand for an item that has no historical demand, known as the cold start problem.

-------------------------------------------------------------------------------------
Question 125:
A data scientist uses an Amazon SageMaker notebook instance to conduct data exploration and analysis. This requires certain 
Python packages that are not natively available on Amazon SageMaker to be installed on the notebook instance. How can a machine 
learning specialist ensure that required packages are automatically available on the notebook instance for the data scientist to use?

    Install AWS Systems Manager Agent on the underlying Amazon EC2 instance and use Systems Manager Automation to execute the 
      package installation commands.
    Create a Jupyter notebook file (.ipynb) with cells containing the package installation commands to execute and place the 
      file under the /etc/init directory of each Amazon SageMaker notebook instance.
    Use the conda package manager from within the Jupyter notebook console to apply the necessary conda packages to the 
      default kernel of the notebook.
  x Create an Amazon SageMaker lifecycle configuration with package installation commands and assign the lifecycle configuration 
      to the notebook instance.
  
-------------------

Customization of a SageMaker notebook instance using an LCC script
  https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-lifecycle-config.html

  - A lifecycle configuration (LCC) provides shell scripts that run only when you create the notebook instance or whenever 
    you start one. When you create a notebook instance, you can create a new LCC or attach an LCC that you already have. 
  - Lifecycle configuration scripts are useful for the following use cases:

      - Installing packages or sample notebooks on a notebook instance

      - Configuring networking and security for a notebook instance

      - Using a shell script to customize a notebook instance

  - You can also use a lifecycle configuration script to access AWS services from your notebook. 
  - For example, you can create a script that lets you use your notebook to control other AWS resources, such as an Amazon EMR instance.


-------------------------------------------------------------------------------------
Question 126:
A data scientist needs to identify fraudulent user accounts for a company's ecommerce platform. The company wants the ability to 
determine if a newly created account is associated with a previously known fraudulent user. The data scientist is using AWS Glue 
to cleanse the company's application logs during ingestion. Which strategy will allow the data scientist to identify fraudulent accounts?

    Execute the built-in FindDuplicates Amazon Athena query.
  x Create a FindMatches machine learning transform in AWS Glue.
    Create an AWS Glue crawler to infer duplicate accounts in the source data.
    Search for duplicate accounts in the AWS Glue Data Catalog.

-------------------

AWS Glue now provides FindMatches ML transform to deduplicate and find matching records in your dataset
  https://aws.amazon.com/about-aws/whats-new/2019/08/aws-glue-provides-findmatches-ml-transform-to-deduplicate/

Posted On: Aug 9, 2019

  - You can now use AWS Glue to find matching records across a dataset (including ones without identifiers) by using 
    the new FindMatches ML Transform, a custom machine learning transformation that helps you identify matching records. 
  - By adding the FindMatches transformation to your Glue ETL jobs, you can find related products, places, suppliers, customers, 
    and more.

  - You can also use the FindMatches transformation for deduplication, such as to identify customers who have signed up more 
    than once, products that have accidentally been added to your product catalog more than once, and so forth. 
  - You can teach the FindMatches ML Transform your definition of a “duplicate” through examples, and it will use machine learning 
    to identify other potential duplicates in your dataset.

-------------------------------------------------------------------------------------
Question 127:
A data scientist has developed a machine learning translation model for English to Japanese by using Amazon SageMaker's 
built-in seq2seq algorithm with 500,000 aligned sentence pairs. While testing with sample sentences, the data scientist 
finds that the translation quality is reasonable for an example as short as five words. However, the quality becomes 
unacceptable if the sentence is 100 words long. Which action will resolve the problem?

    Change preprocessing to use n-grams.
    Add more nodes to the Recurrent Neural Network (RNN) than the largest sentence's word count.
  x Adjust hyperparameters related to the attention mechanism.
    Choose a different weight initialization type.

-------------------
How Sequence-to-Sequence Works
  https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq-howitworks.html

Attention mechanism. 
  - The disadvantage of an encoder-decoder framework is that model performance decreases as and when the length of the 
    source sequence increases because of the limit of how much information the fixed-length encoded feature vector can contain. 
  - To tackle this problem, in 2015, Bahdanau et al. proposed the attention mechanism. 
  - In an attention mechanism, the decoder tries to find the location in the encoder sequence where the most important 
    information could be located and uses that information and previously decoded words to predict the next token in the sequence. 

-------------------

SageMaker -> Developer Guide -> Sequence-to-Sequence Hyperparameters
  https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq-hyperparameters.html


  [attention mechanism related hyperparameters]

  encoder_type 	
     - Encoder type. The 'rnn' architecture is based on attention mechanism by Bahdanau et al. and 'cnn' architecture 
       is based on Gehring et al.
     - Optional; Valid values: String. Either 'rnn' or 'cnn'.; Default value: rnn

  rnn_attention_in_upper_layers 	
    - Pass the attention to upper layers of 'rnn', like Google NMT paper. Only applicable if more than one layer is used.
    - Optional;  Valid values: boolean (true or false);  Default value: 'true'

  rnn_attention_num_hidden 	
    - Number of hidden units for attention layers. defaults to rnn_num_hidden.
    - Optional;  Valid values: positive integer;  Default value: rnn_num_hidden

  rnn_attention_type 	
    - Attention model for encoders. 'mlp' refers to concat and 'bilinear' refers to general from the Luong et al. paper.
    - Optional;  Valid values: String. One of 'dot', 'fixed', 'mlp', or 'bilinear'.;  Default value: mlp

-------------------------------------------------------------------------------------
Question 131:
A company is using Amazon Textract to extract textual data from thousands of scanned text-heavy legal documents daily. The company 
uses this information to process loan applications automatically. Some of the documents fail business validation and are returned 
to human reviewers, who investigate the errors. This activity increases the time to process the loan applications. What should the 
company do to reduce the processing time of loan applications?

    Configure Amazon Textract to route low-confidence predictions to Amazon SageMaker Ground Truth. Perform a manual review on 
      those words before performing a business validation.
    Use an Amazon Textract synchronous operation instead of an asynchronous operation.
 x  Configure Amazon Textract to route low-confidence predictions to Amazon Augmented AI (Amazon A2I). Perform a manual review 
      on those words before performing a business validation.
    Use Amazon Rekognition's feature to detect text in an image to extract the data from scanned images. Use this information 
      to process the loan applications.

-------------------

Using Amazon Augmented AI for Human Review
  https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-use-augmented-ai-a2i-human-review-loops.html

  - When you use AI applications such as Amazon Rekognition, Amazon Textract, or your custom machine learning (ML) models, 
    you can use Amazon Augmented AI to get human review of low-confidence predictions or random prediction samples.

  What is Amazon Augmented AI?
    - Amazon Augmented AI (Amazon A2I) is a service that brings human review of ML predictions to all developers by removing 
      the heavy lifting associated with building human review systems or managing large numbers of human reviewers. 

-------------------------------------------------------------------------------------
Question 134:
A financial services company wants to adopt Amazon SageMaker as its default data science environment. The company's data 
scientists run machine learning (ML) models on confidential financial data. The company is worried about data egress and wants 
an ML engineer to secure the environment. Which mechanisms can the ML engineer use to control data egress from SageMaker? 
(Choose three.)

  x Connect to SageMaker by using a VPC interface endpoint powered by AWS PrivateLink.
    Use SCPs to restrict access to SageMaker.
    Disable root access on the SageMaker notebook instances.
  x Enable network isolation for training jobs and models.
    Restrict notebook presigned URLs to specific IPs used by the company.
  x Protect data with encryption at rest and in transit. Use AWS Key Management Service (AWS KMS) to manage encryption keys.


-------------------
AWS -> Documentation -> Amazon SageMaker -> Developer Guide -> Run Training and Inference Containers in Internet-Free Mode
  https://docs.aws.amazon.com/sagemaker/latest/dg/mkt-algo-model-internet-free.html

  - SageMaker training and deployed inference containers are internet-enabled by default. 
  - This allows containers to access external services and resources on the public internet as part of your training 
    and inference workloads. 

  - If you do not want SageMaker to provide external network access to your training or inference containers, you can 
    enable network isolation.

  Network Isolation
   - You can enable network isolation when you create your training job or model by setting the value of the 
     EnableNetworkIsolation parameter to True when you call CreateTrainingJob, CreateHyperParameterTuningJob, or CreateModel. 

   - If you enable network isolation, the containers can't make any outbound network calls, even to other AWS services such 
     as Amazon S3. 
   - Additionally, no AWS credentials are made available to the container runtime environment. 
   - In the case of a training job with multiple instances, network inbound and outbound traffic is limited to the peers of 
     each training container. 
   - SageMaker still performs download and upload operations against Amazon S3 using your SageMaker execution role in 
     isolation from the training or inference container. 

-------------------------------------------------------------------------------------
Question 136:
A company is converting a large number of unstructured paper receipts into images. The company wants to create a model based on 
natural language processing (NLP) to find relevant entities such as date, location, and notes, as well as some custom entities 
such as receipt numbers. The company is using optical character recognition (OCR) to extract text for data labeling. However, 
documents are in different structures and formats, and the company is facing challenges with setting up the manual workflows for 
each document type. Additionally, the company trained a named entity recognition (NER) model for custom entity detection using a 
small sample size. This model has a very low confidence score and will require retraining with a large dataset. Which solution for 
text extraction and entity detection will require the LEAST amount of effort?

    Extract text from receipt images by using Amazon Textract. Use the Amazon SageMaker BlazingText algorithm to train on the 
      text for entities and custom entities.
    Extract text from receipt images by using a deep learning OCR model from the AWS Marketplace. Use the NER deep learning 
      model to extract entities.
  x Extract text from receipt images by using Amazon Textract. Use Amazon Comprehend for entity detection, and use Amazon 
      Comprehend custom entity recognition for custom entity detection.
    Extract text from receipt images by using a deep learning OCR model from the AWS Marketplace. Use Amazon Comprehend for 
      entity detection, and use Amazon Comprehend custom entity recognition for custom entity detection.

-------------------

Developer Guide -> Amazon Comprehend
  https://docs.aws.amazon.com/comprehend/latest/dg/how-it-works.html

  How it works
    - Amazon Comprehend uses a pre-trained model to gather insights about a document or a set of documents. 
    - This model is continuously trained on a large body of text so that there is no need for you to provide training data.

    - You can use Amazon Comprehend to build your own custom models for custom classification and custom entity recognition. 

    - Amazon Comprehend provides topic modeling using a built-in model. 
    - Topic modeling examines a corpus of documents and organizes the documents based on similar keywords within them.

    - Amazon Comprehend provides synchronous and asynchronous document processing modes. 
    - Use synchronous mode for processing one document or a batch of up to 25 documents. 
    - Use an asynchronous job to process a large number of documents.

  Insights:
    - Amazon Comprehend can analyze a document or set of documents to gather insights about it. 
    - Some of the insights that Amazon Comprehend develops about a document include:

      Entities 
        – Amazon Comprehend returns a list of entities, such as people, places, and locations, identified in a document.
      Events 
        – Amazon Comprehend detects speciﬁc types of events and related details.
      Key phrases 
        – Amazon Comprehend extracts key phrases that appear in a document. 
        - For example, a document about a basketball game might return the names of the teams, the name of the venue, and the final score.
      Personally identifiable information (PII) 
        – Amazon Comprehend analyzes documents to detect personal data that identify an individual, such as an address, 
          bank account number, or phone number.
      Dominant language 
        – Amazon Comprehend identifies the dominant language in a document. Amazon Comprehend can identify 100 languages.
      Sentiment 
        – Amazon Comprehend determines the dominant sentiment of a document. 
        - Sentiment can be positive, neutral, negative, or mixed.
      Targeted Sentiment 
        – Amazon Comprehend determines the sentiment of specific entities mentioned in a document. 
        - The sentiment of each mention can be positive, neutral, negative, or mixed.
      Syntax analysis 
        – Amazon Comprehend parses each word in your document and determines the part of speech for the word. 
        - For example, in the sentence "It is raining today in Seattle," "it" is identified as a pronoun, "raining" is 
          identified as a verb, and "Seattle" is identified as a proper noun.
    
    Amazon Comprehend Custom
      - You can customize Amazon Comprehend for your specific requirements without the skillset required to build machine 
        learning-based NLP solutions. Using automatic machine learning, or AutoML, Comprehend Custom builds customized NLP models 
        on your behalf, using training data that you provide.

      Input document processing 
        – Amazon Comprehend supports one-step document processing for custom classification and custom entity recognition. 
        - For example, you can input a mix of plain text documents and semi-structured documents (such as PDF documents, Microsoft 
          Word documents, and images) to a custom analysis job. 

      Custom classification 
        – Create custom classification models (classifiers) to organize your documents into your own categories. 
        - For each classification label, provide a set of documents that best represent that label and train your classifier on it. 
        - Once trained, a classifier can be used on any number of unlabeled document sets. 
        - You can use the console for a code-free experience or install the latest AWS SDK. 

      Custom entity recognition 
        – Create custom entity recognition models (recognizers) that can analyze text for your specific terms and noun-based phrases. 
        - You can train recognizers to extract terms like policy numbers, or phrases that imply a customer escalation. 
        - To train the model, you provide a list of the entities and a set of documents that contain them. 
        - Once the model is trained, you can submit analysis jobs against it to extract their custom entities.

-------------------------------------------------------------------------------------
Question 138:
A data scientist has been running an Amazon SageMaker notebook instance for a few weeks. During this time, a new version of 
Jupyter Notebook was released along with additional software updates. The security team mandates that all running SageMaker notebook 
instances use the latest security and software updates provided by SageMaker. How can the data scientist meet this requirements?

    Call the CreateNotebookInstanceLifecycleConfig API operation.
    Create a new SageMaker notebook instance and mount the Amazon Elastic Block Store (Amazon EBS) volume from the original instance.
  x Stop and then restart the SageMaker notebook instance.
    Call the UpdateNotebookInstanceLifecycleConfig API operation.

-------------------

SageMaker -> Developer Guide -> Notebook Instance Software Updates
  https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-software-updates.html

 Amazon SageMaker periodically tests and releases software that is installed on notebook instances. This includes:
  - Kernel updates
  - Security patches
  - AWS SDK updates
  - Amazon SageMaker Python SDK
  - updates
  - Open source software updates

 To ensure that you have the most recent software updates, stop and restart your notebook instance, either in the SageMaker 
   console or by calling 'StopNotebookInstance'.

  You can also manually update software installed on your notebook instance while it is running by using update commands in a 
    terminal or in a notebook.

-------------------------------------------------------------------------------------
-------------------
-------------------------------------------------------------------------------------
-------------------
-------------------------------------------------------------------------------------
