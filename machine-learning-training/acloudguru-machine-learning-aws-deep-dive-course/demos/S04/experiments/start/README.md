------------------------------------------------------
4.6 Using SageMaker Experiments To Manage Training Jobs

  Cost: $1 USD

  Machine Learning is an Iterative Process

         Training  -------------> Evaluation ---------------> Prediction
             |                        |
             |            Do we need to change the algorithm?
             |            Do we need to do more feature engineering?
             |            Do we need new or different data?
             |                        |
             |------------------------|

    - over time, you may have thousands of similar training jobs

  Why SageMaker Experiments
    - a capability of SageMaker thats lets you organize, track, compare, and evaluate your
      ML experiments
    - Created through the SageMaker Python SDK
    - track and visualize [the experiments] through the SageMaker UI

  The Experiment Hierarchy
     - an Experiment is a collection of 1 or more Runs
     - a "Run" consists of all the inputs, parameters, configurations for one iteration of training of the model

                      Experiment
                            |
           |-------------|---------|-|-|-------|
           |             |         | | |       |
          Run 1         Run 2      . . .    Run N


  Creating an Experiment through the Python SDK
    - reviewing code in the SageMaker Studio Notebook
         - Dataset using MNIST Dataset with Keras
         - The MNIST database (Modified National Institute of Standards and Technology database[1]) is a large
           database of handwritten digits that is commonly used for training various image processing systems
    - creating multiple runs


      # using the SageMaker Studio Jupyter Notebook:
        -> AWS -> SageMaker -> Domains -> Select [click in to] domain
        # "experiments_keras.ipynb" uses images from cancer flow
        ->  right click on "Launch" for selected user profile -> Studio
                 -> Home <left tab> -> Folder -> S04/experiments/start/experiments_keras.ipynb <double click>
                  <defaults> -> select

                  -> No Kernel <upper right> -> Image: TensorFlow 2.6 Python 3.8 CPU optimized,
                     kernel: Python 3, instance type: ml.t3.medium -> Select
         # in Jupyter Notebook:
           Initialize Environment and Variables:
             install required packages (boto3, sagemaker, tensorflow)
             - set up sagemaker env
             - prepare the data from training

           Data
             - download dataset
             # spit 33% train, 66% validation
             x_train shape: (60000, 28, 28, 1)
             60000 train samples
             10000 test samples
            Build the Model
            Define Keras Callbacks

              - The Keras Callback class provides a method on_epoch_end, which emits metrics at the end of each epoch.
              - All emitted metrics will be logged in the run passed to the callback.
            Set up a SageMaker Experiment and its Runs, then Train
               - train the Keras model locally on the same instance where this notebook is running. With each run,
                 we track the input artifacts and write them to files. We use the ExperimentCallback method to log the
                 metrics to the Experiment run.

               - define experiment and define run name
               experiment_name = "mnist-keras-experiment"
               run_name = "mnist-keras-batch-size-10"

               - will do 1 run with batch_size set to 10 and 2nd run with set to 20
                 - each run will have 3 epoch
                 set:
                   batch_size = 10

++++++++++++++++++
"Set up a SageMaker Experiment and its Runs, then Train" code:
+++++++++++++++++++++++++++++
from sagemaker.experiments.run import Run

batch_size = 10
epochs = 3
dropout = 0.5

model = get_model(dropout)

experiment_name = "mnist-keras-experiment"
run_name = "mnist-keras-batch-size-10"
with Run(experiment_name=experiment_name, run_name=run_name, sagemaker_session=sagemaker_session) as run:
    run.log_parameter("batch_size", batch_size)
    run.log_parameter("epochs", epochs)
    run.log_parameter("dropout", dropout)

    run.log_file("datasets/input_train.npy", is_output=False)
    run.log_file("datasets/input_test.npy", is_output=False)
    run.log_file("datasets/input_train_labels.npy", is_output=False)
    run.log_file("datasets/input_test_labels.npy", is_output=False)

    # Train locally
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[ExperimentCallback(run, model, x_test, y_test)],
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    run.log_metric(name="Final Test Loss", value=score[0])
    run.log_metric(name="Final Test Accuracy", value=score[1])
+++++++++++++++++++++++++++++

  Viewing and Comparing Experiment Results in the SageMaker UI
    - Viewing experiments
    - viewing runs
    - Comparing Runs

      -> SageMaker Studio -> Home <left tab> -> Experiments -> mnist-keras-experiment ->

      # to compare -> select both runs -> analyze

      # to chart
        -> Chart -> confortable -> edit <icon> -> chart type: bar, y-axis: try different values


  Deleting Experiments through the Python SDK
   https://docs.aws.amazon.com/sagemaker/latest/dg/experiments-cleanup.html
   - no way to delete experiment via UI, so must use SDK
++++++
delete experiment SDK code (Note: change "_Experiment" to "Experiment"
++++++++++++++++++++++++++
# Delete the experiment
from sagemaker.experiments.experiment import Experiment

exp = Experiment.load(experiment_name=experiment_name, sagemaker_session=sagemaker_session)
exp._delete_all(action="--force")
++++++++++++++++++++++++++

   - terminate instances and kernels

  Summary

    SageMaker Experiments
      - Help manage the iterative nature of machine learning
         - lets you organize, track, and compare different training models
      - Create through the SageMaker Python SDK
      - View and compare through the SageMaker UI
      - Delete experiments through the SageMaker Python SDK (cannot be deleted in UI)




