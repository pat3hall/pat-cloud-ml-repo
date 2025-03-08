#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U sagemaker')


# In[2]:


import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor

region = boto3.session.Session().region_name

role = get_execution_role()
sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0", role=role, instance_type="ml.m5.xlarge", instance_count=1
)


# In[3]:


import pandas as pd

input_data = "s3://sagemaker-sample-data-{}/processing/census/census-income.csv".format(region)
df = pd.read_csv(input_data, nrows=10)
df.head(n=10)


# In[4]:


get_ipython().run_cell_magic('writefile', 'preprocessing.py', '\nimport argparse\nimport os\nimport warnings\n\nimport pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.compose import make_column_transformer\n\nfrom sklearn.exceptions import DataConversionWarning\n\nwarnings.filterwarnings(action="ignore", category=DataConversionWarning)\n\n\ncolumns = [\n    "age",\n    "education",\n    "major industry code",\n    "class of worker",\n    "num persons worked for employer",\n    "capital gains",\n    "capital losses",\n    "dividends from stocks",\n    "income",\n]\nclass_labels = [" - 50000.", " 50000+."]\n\n\ndef print_shape(df):\n    negative_examples, positive_examples = np.bincount(df["income"])\n    print(\n        "Data shape: {}, {} positive examples, {} negative examples".format(\n            df.shape, positive_examples, negative_examples\n        )\n    )\n\n\nif __name__ == "__main__":\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)\n    args, _ = parser.parse_known_args()\n\n    print("Received arguments {}".format(args))\n\n    input_data_path = os.path.join("/opt/ml/processing/input", "census-income.csv")\n\n    print("Reading input data from {}".format(input_data_path))\n    df = pd.read_csv(input_data_path)\n    df = pd.DataFrame(data=df, columns=columns)\n    df.dropna(inplace=True)\n    df.drop_duplicates(inplace=True)\n    df.replace(class_labels, [0, 1], inplace=True)\n\n    negative_examples, positive_examples = np.bincount(df["income"])\n    print(\n        "Data after cleaning: {}, {} positive examples, {} negative examples".format(\n            df.shape, positive_examples, negative_examples\n        )\n    )\n\n    split_ratio = args.train_test_split_ratio\n    print("Splitting data into train and test sets with ratio {}".format(split_ratio))\n    X_train, X_test, y_train, y_test = train_test_split(\n        df.drop("income", axis=1), df["income"], test_size=split_ratio, random_state=0\n    )\n\n    preprocess = make_column_transformer(\n        (\n            ["age", "num persons worked for employer"],\n            KBinsDiscretizer(encode="onehot-dense", n_bins=10),\n        ),\n        (["capital gains", "capital losses", "dividends from stocks"], StandardScaler()),\n        (["education", "major industry code", "class of worker"], OneHotEncoder(sparse=False)),\n    )\n    print("Running preprocessing and feature engineering transformations")\n    train_features = preprocess.fit_transform(X_train)\n    test_features = preprocess.transform(X_test)\n\n    print("Train data shape after preprocessing: {}".format(train_features.shape))\n    print("Test data shape after preprocessing: {}".format(test_features.shape))\n\n    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")\n    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")\n\n    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")\n    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")\n\n    print("Saving training features to {}".format(train_features_output_path))\n    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)\n\n    print("Saving test features to {}".format(test_features_output_path))\n    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)\n\n    print("Saving training labels to {}".format(train_labels_output_path))\n    y_train.to_csv(train_labels_output_path, header=False, index=False)\n\n    print("Saving test labels to {}".format(test_labels_output_path))\n    y_test.to_csv(test_labels_output_path, header=False, index=False)\n')


# In[5]:


from sagemaker.processing import ProcessingInput, ProcessingOutput

sklearn_processor.run(
    code="preprocessing.py",
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
    ],
    arguments=["--train-test-split-ratio", "0.2"],
)

preprocessing_job_description = sklearn_processor.jobs[-1].describe()

output_config = preprocessing_job_description["ProcessingOutputConfig"]
for output in output_config["Outputs"]:
    if output["OutputName"] == "train_data":
        preprocessed_training_data = output["S3Output"]["S3Uri"]
    if output["OutputName"] == "test_data":
        preprocessed_test_data = output["S3Output"]["S3Uri"]


# In[ ]:




