

tutorial: End-to-End Example with Amazon SageMaker K-Means
   https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-python-sdk/1P_kmeans_highlevel/kmeans_mnist.html

github:
  https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/1P_kmeans_highlevel/kmeans_mnist.ipynb

    Code: kmeans_mnist.ipynb


        >>> # # End-to-End Example with Amazon SageMaker K-Means
        >>> #

        >>> #
        >>> # 1. [Introduction](#Introduction)
        >>> # 2. [Prerequisites and Preprocessing](#Prequisites-and-Preprocessing)
        >>> #   1. [Permissions and environment variables](#Permissions-and-environment-variables)
        >>> #   2. [Data ingestion](#Data-ingestion)
        >>> #   3. [Data inspection](#Data-inspection)
        >>> #   4. [Data conversion](#Data-conversion)
        >>> # 3. [Training the K-Means model](#Training-the-K-Means-model)
        >>> # 4. [Set up hosting for the model](#Set-up-hosting-for-the-model)
        >>> # 5. [Validate the model for use](#Validate-the-model-for-use)
        >>> #

        >>> # ## Introduction
        >>> #
        >>> # Welcome to our first end-to-end example! Today, we're working through a classification problem, specifically of images
        >>> # of handwritten digits, from zero to nine. Let's imagine that this dataset doesn't have labels, so we don't know for sure
        >>> # what the true answer is. In later examples, we'll show the value of "ground truth", as it's commonly known.
        >>> #
        >>> # Today, however, we need to get these digits classified without ground truth. A common method for doing this is a set of
        >>> # methods known as "clustering", and in particular, the method that we'll look at today is called k-means clustering. In this
        >>> # method, each point belongs to the cluster with the closest mean, and the data is partitioned into a number of clusters that
        >>> # is specified when framing the problem. In this case, since we know there are 10 clusters, and we have no labeled data (in
        >>> # the way we framed the problem), this is a good fit.
        >>> #
        >>> # To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on.

        >>> # ## Prequisites and Preprocessing
        >>> #
        >>> # This notebook was tested in Amazon SageMaker Studio on a ml.t3.medium instance with Python 3 (Data Science) kernel.
        >>> #
        >>> # ### Permissions and environment variables
        >>> #
        >>> # Here we set up the linkage and authentication to AWS services. There are two parts to this:
        >>> #
        >>> # 1. The role(s) used to give learning and hosting access to your data. Here we extract the role you created earlier
        >>> #    for accessing your notebook. See the documentation if you want to specify a different role.
        >>> # 1. The S3 bucket name that you want to use for training and model data. Here we use a default in the form of
        >>> #    `sagemaker-{region}-{AWS account ID}`, but you may specify a different one if you wish.

        >>> !matplotlib', 'inline


        >>> # Let's make sure we have the latest version of the SageMaker Python SDK

        >>> !pip install --upgrade sagemaker


        >>> from sagemaker import get_execution_role
        >>> from sagemaker.session import Session
        >>> from sagemaker.utils import S3DataConfig

        >>> role = get_execution_role()
        >>> sm_session = Session()
        >>> bucket = sm_session.default_bucket()


        >>> # ### Data ingestion
        >>> #
        >>> # Next, we read the dataset from the existing repository into memory, for preprocessing prior to training.  In this case we'll
        >>> # use the MNIST dataset [1], which contains 70K 28 x 28 pixel images of handwritten digits.  For more details, please see
        >>> # [here](http://yann.lecun.com/exdb/mnist/).
        >>> #
        >>> # This processing could be done *in situ* by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc., assuming the
        >>> # dataset is present in the appropriate location. Then, the next step would be to transfer the data to S3 for use in training.
        >>> # For small datasets, such as this one, reading into memory isn't onerous, though it would be for larger datasets.
        >>> #

        >>> import pickle, gzip, numpy, boto3, json

        >>> # Load the dataset
        >>> s3 = boto3.client("s3")
        >>> data_bucket = S3DataConfig(
        >>>     sm_session, "example-notebooks-data-config", "config/data_config.json"
        >>> ).get_data_bucket()
        >>> print(f"Using data from {data_bucket}")
            Using data from sagemaker-sample-files

            # extract the train dataset, valid  dataset and test dataset
        >>> s3.download_file(data_bucket, "datasets/image/MNIST/mnist.pkl.gz", "mnist.pkl.gz")
        >>> with gzip.open("mnist.pkl.gz", "rb") as f:
        >>>     train_set, valid_set, test_set = pickle.load(f, encoding="latin1")


        >>> # ### Data inspection
        >>> #
        >>> # Once the dataset is imported, it's typical as part of the machine learning process to inspect the data, understand the
        >>> # distributions, and determine what type(s) of preprocessing might be needed. You can perform those tasks right here in
        >>> # the notebook. As an example, let's go ahead and look at one of the digits that is part of the dataset.

        >>> import matplotlib.pyplot as plt

            # set figure size to 2 x 10 inches
        >>> plt.rcParams["figure.figsize"] = (2, 10)


        >>> def show_digit(img, caption="", subplot=None):
        >>>     if subplot is None:
        >>>         _, (subplot) = plt.subplots(1, 1)
            # MNIST are 28x28 -> not sure why to reshape
        >>>     imgr = img.reshape((28, 28))
        >>>     subplot.axis("off")
        >>>     subplot.imshow(imgr, cmap="gray")
        >>>     plt.title(caption)


            # plot train image 30 and display it label
        >>> show_digit(train_set[0][30], f"This is a {train_set[1][30]}")

            # train_set is a tuple that contains 50000 28x28 (784) images at train_set[0] and 50000 integer labesl at train_set[1]
        >>> len(train_set)
            2
        >>> len(train_set[0])
            50000
        >>> len(train_set[0][1]])
            728
        >>> len(train_set[1])
            50000

            # valid_set is a tuple that contains 10000 28x28 (784) images at valid_set[0] and 50000 integer labesl at valid_set[1]
            # test_set  is a tuple that contains 10000 28x28 (784) images at test_set[0]  and 50000 integer labesl at test_set[1]


        >>> # ## Training the K-Means model
        >>> #
        >>> # Once we have the data preprocessed and available in the correct format for training, the next step is to actually train the
        >>> # model using the data. Since this data is relatively small, it isn't meant to show off the performance of the k-means training
        >>> # algorithm.  But Amazon SageMaker's k-means has been tested on, and scales well with, multi-terabyte datasets.
        >>> #
        >>> # After setting training parameters, we kick off training, and poll for status until training is completed, which in this example,
        >>> # takes around 4 minutes.

        >>> from sagemaker import KMeans

        >>> data_location = f"s3://{bucket}/kmeans_highlevel_example/data"
        >>> output_location = f"s3://{bucket}/kmeans_example/output"

        >>> print(f"training data will be uploaded to: {data_location}")
        >>> print(f"training artifacts will be uploaded to: {output_location}")

            # K-means: The Amazon SageMaker K-means algorithm.
            #  class sagemaker.KMeans(
            # https://sagemaker.readthedocs.io/en/v2.29.2/algorithms/kmeans.html
            # use sagemaker.KMeans() to create Kmeans Estimator which mostly uses default hyperparameters setting
        >>> kmeans = KMeans(
        >>>     role=role,
        >>>     instance_count=2,
        >>>     instance_type="ml.c4.xlarge",
        >>>     output_path=output_location,
        >>>     k=10,
        >>>     data_location=data_location,
        >>> )
            training data will be uploaded to: s3://sagemaker-us-east-2-012345678901/kmeans_highlevel_example/data
            training artifacts will be uploaded to: s3://sagemaker-us-east-2-012345678901/kmeans_example/output


            # train Kmeans model with fit()
            #  record_set(): to construct a RecordSet object from ndarray arrays.
        >>> %%time
            nkmeans.fit(kmeans.record_set(train_set[0]))


        >>> # ## Set up hosting for the model
        >>> # Now, we can deploy the model we just trained behind a real-time hosted endpoint.  This next step can take, on average,
            # 7 to 11 minutes to complete.

            # deploy model
        >>> %%time
        >>> kmeans_predictor = kmeans.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
            INFO:sagemaker.image_uris:Same images used for training and inference. Defaulting to image scope: inference.
            INFO:sagemaker.image_uris:Ignoring unnecessary instance type: None.
            INFO:sagemaker:Creating model with name: kmeans-2024-10-26-01-12-04-799
            INFO:sagemaker:Creating endpoint-config with name kmeans-2024-10-26-01-12-04-799
            INFO:sagemaker:Creating endpoint with name kmeans-2024-10-26-01-12-04-799

            -------!CPU times: user 105 ms, sys: 8.35 ms, total: 113 ms
            Wall time: 4min 1s


        >>> # ## Validate the model for use
        >>> # Finally, we'll validate the model for use. Let's generate a classification for a single observation from the trained
        >>> # model using the endpoint we just created.

        >>> result = kmeans_predictor.predict(train_set[0][30:31])
        >>> print(result)
            [label {
              key: "distance_to_cluster"
              value {
                float32_tensor {
                  values: 7.27719927
                }
              }
            }
            label {
              key: "closest_cluster"
              value {
                float32_tensor {
                  values: 6
                }
              }
            }
            ]


        >>> train_set[1][30:31]
             array([3])
        >>> # OK, a single prediction works.

            # Actually, the prediction did NOT work - it predicted '6', but it was actually '3'. However, the images shows
            #  the bottom loop was mostly closed

            # display image
        >>> show_digit(train_set[0][30], f"This is a {train_set[1][30]}")

        >>> # Let's do a whole batch and see how well the clustering works.

            # predict class/cluster for 1st 100 images in valid set and extract predicted values
        >>> %%time
            result = kmeans_predictor.predict(valid_set[0][0:100])
            clusters = [r.label["closest_cluster"].float32_tensor.values[0] for r in result]


            # print images in each of the 10 predicted clusters
        >>> for cluster in range(10):
        >>>     print(f"\n\n\nCluster {int(cluster)}:")
        >>>     digits = [img for l, img in zip(clusters, valid_set[0]) if int(l) == cluster]
        >>>     """
        >>>         The KMeans algorithm as an optimization problem is an NP Complete problem, and internal implementations
        >>>         can produce different results for each run, depending upon the locations of the initial cluster centroid.
        >>>         In some cases, there might be no data points in a cluster. We plot below the data points for clusters which
        >>>         have datapoints.
        >>>     """
        >>>     if digits:
        >>>         height = ((len(digits) - 1) // 5) + 1
        >>>         width = 5
        >>>         plt.rcParams["figure.figsize"] = (width, height)
        >>>         _, subplots = plt.subplots(height, width)
        >>>         subplots = numpy.ndarray.flatten(subplots)
        >>>         for subplot, image in zip(subplots, digits):
        >>>             show_digit(image, subplot=subplot)
        >>>         for subplot in subplots[len(digits) :]:
        >>>             subplot.axis("off")
        >>>         plt.show()

            # all cluster had a mix images with different numbers (expect cluster 2 had all '0'


        >>> # ### The bottom line
        >>> #
        >>> # K-Means clustering is not the best algorithm for image analysis problems, but we do see pretty reasonable clusters being built.

        >>> # ### (Optional) Delete the Endpoint
        >>> # If you're ready to be done with this notebook, make sure run the cell below.  This will remove the hosted endpoint
        >>> #  you created and avoid any charges from a stray instance being left on.

        >>> print(kmeans_predictor.endpoint)
            WARNING:sagemaker.deprecations:The endpoint attribute has been renamed in sagemaker>=2.
             See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.

        >>> kmeans_predictor.delete_endpoint()
            INFO:sagemaker:Deleting endpoint configuration with name: kmeans-2024-10-26-01-12-04-799
            INFO:sagemaker:Deleting endpoint with name: kmeans-2024-10-26-01-12-04-799


