------------------------------------------------------
2.8 DEMO: Using Docker Containers with Amazon SageMaker




  using Docker containers with Amazon SageMaker.
    - SageMaker supports containerization for both training and inference.
    - if there's no prebuilt container for your use case, create your own
      - SageMaker enables you to package your own algorithms in a Docker container that can then be trained
        and deployed in the SageMaker environment.
   demo:
     - in  an AI sandbox
     - build a Docker container,
     - use SageMaker to train, deploy, and evaluate the model.


  ---------------
  Demo

    -> SageMaker AI -> Applications and IDEs -> Notebook -> Create Notebook ->
      Notebook Instance Name: myNotebook , IAM Role: create role,
      -> Create Notebook Instance

      Note: Demo does not work in AWS sandbox -> can not create a SageMaker notebook

      -> Start "my-notebook-inst" -> open jupyterLab notebook

         # download git repo:
         git -> clone a repo -> Git repo URL: https://github.com/pluralsight-cloud/mls-c01-aws-certified-machine-learning-implementation-operations.git ,  unselect "Open Readm files" -> clone

          <under files> -> click on  using-docker-containers-with-sagemaker-demo/docker-containers-with-sagemaker.ipynb



       -> demo related files are provided under demos/2_8_using-docker-containers-with-sagemaker-demo/ :
          -> jupyter notebook:
          docker-containers-with-sagemaker.ipynb
          -> extracted python from jupyter notebook:
          docker-containers-with-sagemaker.py
          -> html view from completed jupyter notebook:
          docker-containers-with-sagemaker.py


  ---------------
  Dockerfile:

# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip --no-cache-dir install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas flask gunicorn

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY decision_trees /opt/program
WORKDIR /opt/program

  ---------------


  Note: Git repo
      using-docker-containers-with-sagemaker-demo\container directory is missing "build_and_push.sh" script

  ---------------

    code: docker-containers-with-sagemaker

      >>> # # Using Docker Containers with Amazon SageMaker Demo

      >>> # ---
      >>> #
      >>> # Use this notebook with a SageMaker notebook Jupyter Lab, not using SageMaker Studio.
      >>> #
      >>> # ---

      >>> # SageMaker, enables you to package your own algorithms that can than be trained and deployed in the SageMaker environment.
      >>> #
      >>> # This demo that shows how to build a Docker container for SageMaker and use it for training and inference, if there is
      >>> # no pre-built container matching your requirements that you can use.

      >>> # ## Part 1: Packaging and Uploading your Algorithm for use with Amazon SageMaker

      >>> # ### The parts of the sample container
      >>> #
      >>> # In the `container` directory are all the components you need to package the sample algorithm for SageMager:
      >>> #
      >>> #     .
      >>> #     |-- Dockerfile
      >>> #     |-- build_and_push.sh
      >>> #     `-- decision_trees
      >>> #         |-- nginx.conf
      >>> #         |-- predictor.py
      >>> #         |-- serve
      >>> #         |-- train
      >>> #         `-- wsgi.py
      >>> #
      >>> # * "Dockerfile" describes how to build your Docker container image. More details below.
      >>> # * "build_and_push.sh" is a script that uses the Dockerfile to build your container images and then pushes it to ECR.
      >>> # * "decision_trees" is the directory containing the files that will be installed in the container.
      >>> # * "local_test" is a directory that shows how to test your new container on any computer that can run Docker.
      >>> #
      >>> # The files that we'll put in the container are:
      >>> #
      >>> # * "nginx.conf" is the configuration file for the nginx front-end. Generally, you should be able to take this file as-is.
      >>> # * "predictor.py" is the program that actually implements the Flask web server and the decision tree predictions for
      >>> #                  this app.
      >>> # * "serve" is the program started when the container is started for hosting. It simply launches the gunicorn server which
      >>> #           runs multiple instances of the Flask app defined in `predictor.py`.
      >>> # * "train" is the program that is invoked when the container is run for training. You can modify this program to implement
      >>> #           your training algorithm.
      >>> # * "wsgi.py" is a small wrapper used to invoke the Flask app.

      >>> # ### The Dockerfile
      >>> #
      >>> # Docker uses a simple file called a `Dockerfile` to specify how the image is assembled.
      >>> #
      >>> # SageMaker uses Docker to allow users to train and deploy models, inculding creating your own.
      >>> #
      >>> # The Dockerfile describes the image that we want to build. You can think of it as describing the complete operating
      >>> # system installation of the system that you want to run. A Docker container running is quite a bit lighter than a full
      >>> # operating system, however, because it takes advantage of Linux on the host machine for the basic operations.
      >>> #
      >>> # We'll use a standard Ubuntu installation and install the things needed by our model.
      >>> # Then add the code that implements our specific algorithm to the container and set up the right environment to run under.
      >>> #
      >>> # Let's review the Dockerfile:

      >>> !cat container/Dockerfile


      >>> # ### Building and registering the container
      >>> #
      >>> # Build the container image using `docker build`.
      >>> # Push the container image to ECR using `docker push`.
      >>> #
      >>> # This code looks for an ECR repository in your account. If the repository doesn't exist, the script will create it.

      >>> %%sh

      >>> # The name of our algorithm
      >>> algorithm_name=sagemaker-decision-trees

      >>> cd container

      >>> chmod +x decision_trees/train
      >>> chmod +x decision_trees/serve

      >>> account=$(aws sts get-caller-identity --query Account --output text)

      >>> # Get the region defined in the current configuration (default to us-east-1 if none defined)
      >>> region=$(aws configure get region)
      >>> region=${region:-us-east-1}

      >>> fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

      >>> # If the repository doesn't exist in ECR, create it.
      >>> aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

      >>> if [ $? -ne 0 ]
      >>> then
      >>>     aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
      >>> fi

      >>> # Get the login command from ECR and execute it directly
      >>> aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}

      >>> # Build the docker image locally with the image name and then push it to ECR
      >>> # with the full name.

      >>> docker build -t ${algorithm_name} .
      >>> docker tag ${algorithm_name} ${fullname}

      >>> docker push ${fullname}


      >>> # ## Part 2: Using your Algorithm in Amazon SageMaker
      >>> #
      >>> # Once you have your container packaged, you can use it to train models and use the model for hosting or batch transforms.
      >>> # Let's do that with the algorithm we made above.
      >>> #
      >>> # ## Set up the environment
      >>> #
      >>> # Here we specify a bucket to use and the role that will be used for working with SageMaker.

      >>> # S3 prefix
      >>> prefix = "DEMO-scikit-byo-iris"

      >>> # Define IAM role
      >>> import boto3
      >>> import re

      >>> import os
      >>> import numpy as np
      >>> import pandas as pd
      >>> from sagemaker import get_execution_role

      >>> role = get_execution_role()


      >>> # ## Create the session
      >>> #
      >>> # The session remembers our connection parameters to SageMaker. We'll use it to perform all of our SageMaker operations.

      >>> import sagemaker as sage
      >>> from time import gmtime, strftime

      >>> sess = sage.Session()


      >>> # ## Upload the data for training
      >>> #
      >>> # For the purposes of this example, we're using some the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), which is in the training folder.

      >>> WORK_DIRECTORY = "data"

      >>> data_location = sess.upload_data(WORK_DIRECTORY, key_prefix=prefix)


      >>> # ## Train the Model
      >>> #
      >>> # In order to use SageMaker to train our algorithm, we'll create an `Estimator` that defines how to use the container
      >>> # to train. This includes the configuration we need to invoke SageMaker training:
      >>> #
      >>> # * The "container name". This is defined above.
      >>> # * The "role". As defined above.
      >>> # * The "instance count" The number of EC2 instances to use for training.
      >>> # * The "instance type" Type of EC2 instance to use for training.
      >>> # * The "output path" Where the model artifact will be written.
      >>> # * The "session" is the SageMaker session object defined above.
      >>> #
      >>> # Then we use fit() on the estimator to train against the data that we uploaded above.

      >>> account = sess.boto_session.client("sts").get_caller_identity()["Account"]
      >>> region = sess.boto_session.region_name
      >>> image = "{}.dkr.ecr.{}.amazonaws.com/sagemaker-decision-trees:latest".format(account, region)

      >>> tree = sage.estimator.Estimator(
      >>>     image,
      >>>     role,
      >>>     1,
      >>>     "ml.m5.large",
      >>>     output_path="s3://{}/output".format(sess.default_bucket()),
      >>>     sagemaker_session=sess,
      >>> )

      >>> tree.fit(data_location)


      >>> # ## Deploying the model
      >>> #
      >>> # After training is complete, deploy the model using the `deploy` API call. Provide the instance count, instance type,
      >>> # and optionally serializer and deserializer functions.

      >>> from sagemaker.serializers import CSVSerializer

      >>> predictor = tree.deploy(
      >>>     initial_instance_count=1,
      >>>     instance_type="ml.m5.large",
      >>>     serializer=CSVSerializer()
      >>> )


      >>> # ### Choose some data and use it for a prediction
      >>> #
      >>> # Make sure the model deployed properly by running some predictions, we'll re-use some of the data we used for training,
      >>> # for the purpose of checking that the model successfully deployed.
      >>> #
      >>> # Choose some data and use it for a prediction
      >>> # In order to do some predictions, we'll extract some of the data we used for training and do predictions against it.
      >>> # This is, of course, bad statistical practice, but a good way to see how the mechanism works.

      >>> shape = pd.read_csv("data/iris.csv", header=None)
      >>> shape.sample(3)


      >>> # drop the label column in the training set
      >>> shape.drop(shape.columns[[0]], axis=1, inplace=True)
      >>> shape.sample(3)


      >>> import itertools

      >>> a = [50 * i for i in range(3)]
      >>> b = [40 + i for i in range(10)]
      >>> indices = [i + j for i, j in itertools.product(a, b)]

      >>> test_data = shape.iloc[indices[:-1]]


      >>> # Prediction is as easy as calling predict with the predictor we got back from deploy and the data we want to do
      >>> # predictions with. The serializers take care of doing the data conversions for us.

      >>> print(predictor.predict(test_data.values).decode("utf-8"))


      >>> # ### Optional cleanup
      >>> # When you're done with the endpoint, you'll want to clean it up.

      >>> sage.Session().delete_endpoint(predictor.endpoint)


  ---------------

