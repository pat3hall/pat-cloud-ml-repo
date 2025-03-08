------------------------------------------------------
4.7 Detecting Bias with SageMaker Clarify


  Cost: $1 USD

  ML Perception
    - ML must be correct - It knows more than I do
    ML Bias examples:
      Dissecting racial bias in an algorithm used to manage the health of populations
      https://www.science.org/doi/10.1126/science.aax2342

      Amazon scraps secret AI recruiting tool that showed bias against women
      https://www.reuters.com/article/us-amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G

  Why SageMaker Clarify
    - Detects and measures potential bias using a variety of metrics
    - detects underlining biases in the model and data

  Two Types of ML Bias
    Dataset (pre-training) bias
      - data is imbalanced and doesn't reflect the real world
      - example: loan approval data contains very little data for people who are self-employed but then
        when the bank is using the model in the real world, it has a lot of self-employed people
    Model (post-training) bias
      - bias introduced by the training algorithm
      - example: binary classification algorithm is used to predict fraud or not, but the model was trained
        with data that showed 99% of transactions were not fraud and so the model might have picked up some bias.

  Integration Points
    Using Clarify with other SageMaker Products
      - SageMaker Data Wrangler
        - Use a simple graphical interface
      - SageMaker Data Experiments
        - Get bias results for each experiment
        - Visual results appear alongside other experiment details

  Detecting Pre-training Bias in a Dataset
    - Reviewing the dataset
      - S04/clarify/start/loan_data.csv
        - home loan dataset with features: loan_id,gender,married,dependents,education,self_employed,applicant_income,etc
        - target: approved (Y or N)
        - looking at gender and self_employed for potential bias
    - Creating a new experiment
    - Running the Clarify Processor job
    - Viewing the bias results in the SageMaker UI



      # Create S3 bucket: clarify-loan-approval-pat

      # using the SageMaker Studio Jupyter Notebook:
        -> AWS -> SageMaker -> Domains -> Select [click in to] domain
        # "experiments_keras.ipynb" uses images from cancer flow
        ->  right click on "Launch" for selected user profile -> Studio
                 -> Home <left tab> -> Folder -> S04/clarify/start/clarify-pre-training-bias.ipynb <double click>
                  <defaults> -> select

         # in Jupyter Notebook:
           Initialize Environment and Variables:
             - check sagemaker version:
             2.192.0
            # To use the Experiments functionality in the SageMaker Python SDK, you need to be running at least
            #    SageMaker v2.123.0
            # skip upgrade sagemaker block
             install required packages (sagemaker > 2)
             - set up sagemaker env
             - prepare the data from training

           # Import libraries
            -> first, update bucket name:
            bucket = 'clarify-loan-approval-pat'
            -> correct import "CSVSerializer to:
             from sagemaker.serializers import CSVSerializer

           Data
             - load data from - S04/clarify/start/loan_data.csv
             - define attributes
             - upload dataset to S3

            Clarify and Experiments
              - implement the Clarify code to detect bias in our dataset.
              - It starts with a processor for the job, then we define various configuration parameters.
              - When we run the pre_training_bias job, we hook into our Experiment.

++++++++++++++++++++++++

# Define the processor for the job
clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    sagemaker_session=sess,
    job_name_prefix='clarify-pre-training-bias-detection-job'
)

# Specify the path where the bias report will be saved once complete
bias_report_output_path = 's3://{}/{}/clarify-bias'.format(bucket, prefix)

# Specify the S3 path to our input data
s3_data_input_path='s3://{}/{}'.format(bucket, prefix)

# Specify inputs, outputs, columns and target names
bias_data_config = clarify.DataConfig(
    s3_data_input_path=s3_data_input_path,
    s3_output_path=bias_report_output_path,
    label='approved',
    headers=df.columns.to_list(),
    dataset_type='text/csv',
)

# Specify the configuration of the bias detection job
# For facet_name, we include two sensitive features we want to check for bias: gender and self-employed
# For facet_values_or_threshold, we input the values of potentially disadvantaged groups (gender of 0 = female; self-employed of 1 = self-employed)
bias_config = clarify.BiasConfig(
    label_values_or_threshold=['Y'], # The value that indicates someone received a home loan
    facet_name=['gender', 'self_employed'],
    facet_values_or_threshold=[[0], [1]]
)
# Create an experiment and start a new run
experiment_name = 'loan-approval-experiment'
run_name = 'pre-training-bias'

# Run the bias detection job, associating it with our Experiment
with Run(
    experiment_name=experiment_name,
    run_name=run_name,
    sagemaker_session=sess,
) as run:
    clarify_processor.run_pre_training_bias(
        data_config=bias_data_config,
        data_bias_config=bias_config,
        logs=False,
    )
# Create an experiment and start a new run
experiment_name = 'loan-approval-experiment'
run_name = 'pre-training-bias'

# Run the bias detection job, associating it with our Experiment
with Run(
    experiment_name=experiment_name,
    run_name=run_name,
    sagemaker_session=sess,
) as run:
    clarify_processor.run_pre_training_bias(
        data_config=bias_data_config,
        data_bias_config=bias_config,
        logs=False,
    )
++++++++++++++++++++++++

  Viewing results in S3
    -> S3 -> "clarify-loan-approval-pat" bucket -> under: demo/clarify-bias
     -> created "report.html", "report.ipynb", & "report.pdf"
     -> examine "report.pdf"

  Viewing results in SageMaker Experiments UI:

      -> SageMaker Studio -> Home <left tab> -> Experiments -> loan-approval-experiment -> clarify-pre-training-bias-detection-*
         -> Bias Reports

  Deleting the Experiment

   https://docs.aws.amazon.com/sagemaker/latest/dg/experiments-cleanup.html
   - no way to delete experiment via UI, so must use SDK
++++++
delete experiment SDK code (Note: change "_Experiment" to "Experiment"
+++++++++++++++++++++
from sagemaker.experiments.experiment import Experiment

exp = Experiment.load(experiment_name=experiment_name, sagemaker_session=sess)
exp._delete_all(action="--force")
+++++++++++++++++++++

  Delete Resources
   - terminate instances and kernels

  Summary

    SageMaker Clarify
      - detects and measures potential bias usign a variety of metrics
      - bias can exist in the dataset or in the trained model
      - integrates with Data Wrangler and Experiments making it easy to work with in a graphical way



