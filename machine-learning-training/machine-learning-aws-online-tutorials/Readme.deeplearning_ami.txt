----------------------------------
   AWS Deep Learning AMIs
----------------------------------
-----------------------------------------------------------

Deep Learning on AWS made simple - AWS Online Tech Talks
  Shashank Prasanna
  https://www.youtube.com/watch?v=1zn7UIHrVw0

Best GPUs on AWS for Deep Learning
  Shashank Prasanna
  https://shashankprasanna.com/best-gpus-on-aws-for-deep-learning/

Summary:
  TensorFlow and PyTorch are the two most popular deep learning (DL) frameworks today for developing and running 
  neural networks. These frameworks are easy to get started, but developers and data scientists find it challenging 
  to setup and ensure that DL frameworks include the latest performance optimizations, and scale them to a large 
  number of CPUs and GPUs to deliver faster results. In this session, you’ll learn how to leverage AWS optimized 
  DL framework containers to train quickly and save cost with Amazon SageMaker, and skip the complicated process of 
  managing and optimizing your framework environments.

  Learning Objectives: 
  * Objective 1: Understand and dive into the most popular deep learning frameworks such as TensorFlow and PyTorch
  * Objective 2: Learn how to get started with deep learning frameworks on Amazon SageMaker
  * Objective 3: Evaluate how to train quickly and save cost with Amazon SageMaker by using AWS optimized DL framework 
                 containers



  Simplifying Deep Learning (DL) workflows on AWS:
    - Development experience (IDE)
    - DL Frameworks
    - Infrastructure Compute (CPUs, GPUs) storage


  Simple flow
     user <--> EC2                                 Storage
               GPU                          <--->  Volume
               Deep Learning AMIs (DLAMI)   <--->  File System
                 

   DLAMI
     - include deep learning frameworks like TensorFlow, PyTorch
     - include development environment

     Development Environment
       - replicates local/laptop development experience on AWS
     DL Frameworks
       - ready access to the AWS optimized for ML frameworks with the DLAMI
     Infrastructure
       - wide range of CPU and GPU options on EC2 with multiple storage options
       - upto 8 GPUs on an EC2 instance


  AWS -> EC2 -> Launch instance -> AMI -> search "deep learning" -> select appropriate DL AMI
      -> enable public IP address -> Launch instance

     ssh -i <pem> <ubuntu>@<public IP address> 
     # launch Jupyter server
     $ tmux &
     # create tunnel from your laptop
     ./ec2-tunnel <public IP Address
     # from web browser - connect to Jupyter notebook
     localhost:8888/lab


  Tips
    - use frameworks provided by DLAMI
    - don't pip or conda install DL frameworks as they're not performance optimized for AWS instances



  Fully managed Jupyter based IDE for DL
   SageMaker Studio Lab IDE
     - free option for CPU (8 hr) and GPU (4 hr) at: studiolab.sagemaker.aws
     - login, start runtime, Open Project (Jupyter Notebook)
     - playground for experimenting

   SageMaker Studio IDE
     - paid option, but more complete deep learning end-to-end experience
     - fully managed Jupyter Notebooks
     - easily switch the Compute instances to use

     -> AWS -> SageMaker -> Studio <left tab> -> Create Domain # 1st time only -> Launch

   Tips:
     - use free SageMaker Studio Lab for learning and prototyping
     - use SageMaker Studio for production workflows (experiment tracking, collaboration, scaling, 
       hosting models, deployment). You get end-to-end CI/CD options from MLOps with SageMaker pipelines
     - for your production workflows

  Scaling
    - to speed up training
    - manage multiple experiments

    Migrating flows (e.g. from laptop to Studio)
      - different versions of TensorFlow, different CPU/GPU, Nivida drivers, OS


  Containers for custom ML environments
    - lightweight
    - portable
    - scalable
    - consistent
  Containers can package
    - training code
    - dependencies
    - configurations

    e.g. TensorFlow container image could include: 
         Tensflow 2.4
           - keras, scikit-learn, horovod, pandas, numpy, openmpi, scipy, python, other
         CPU: 
           - libraries: mki
         GPU: 
           - libraries: cuDNN, cuBLAS, NCCL, CUDA toolkit

       GPU drivers container 
         - seperate container could be used for GPU drivers
         

  Prepackaged ML Framework container images
    github.com/aws/deep-learning-containers
    - fully configured and validated
    - includes AWS optimizations for TensorFlow, PyTorch, MXNet, and Hugging Face
    - hosted in Elastic Container Registry (ECR)
    

    A quick guide to managing machine learning experiment
    https://medium.com/towards-data-science/a-quick-guide-to-managing-machine-learning-experiments-af84da6b060b
    or
    https://towardsdatascience.com/a-quick-guide-to-managing-machine-learning-experiments-af84da6b060b

      - each experiment will use 1 GPU instance and run in parallel using ml.p3.2xlarge' instance
        - 'tf_estimator' function loops through the experiments and fully manages running them


  To scale up to get faster results
     Scaling-up approach:
       - use bigger GPU instance and/or if using CPU instance, change to GPU instance
       - simpliest approach, no code modification
     Scaling-out approach:
       - if large dataset, add more instances
       - requires code modifications
     Tips:
       - always scale-up before you scale-out
    
  Distributed Training
     - instead of running batch1 ... batch8 sequentially
     -  run each batch in parallel to multiple workers, but then there is a step of averaging gradients,
        and updating your workers


    local laptop or desktop   ------->     train.py  | Deep Learning   <----  container
      with SageMaker SDK                             | container              Registery

                                                     |
                                                     V
                                      Fully Managed SageMaker Cluster   <-----> S3 (Datasets)

   Distributed Training Code Changes
      - change to estimator function - add a distribution parameter
           instance_type = 'ml.p3.16xlarge'   # 8 Nvidia V100 GPUs
           instance_count = 1
           distribution={'smdistributed': {'dataparallel':{'enabled': True} }}

           # above tells sagemaker to distribute parallel jobs across the 8 GPUs

      - also need to modify your train script code for averaging gradients, etc.
        import smdistributed.dataparallel.tensorflow as smdp

        smdp.init()
        size = smdp.size()   # returns cluster size, e.g. 8 GPUs

        # Training step
        @tf.function
        def training_step(images, labels, first_batch):
            with tf.GradientTape() as tape:
                train_pred = model(images, training=True)
                loss_value = loss(labels, train_pred)

            # change: Wrap tf.gradientTape with SMDataParallel's DistributedGradientTape
            tape = smdp.DistributedGradientTape(tape)
        
            grads = tape.gradient(loss_value,model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            if first_batch:
               # change: Broadcast model and optimizer variables
               smdp.broadcast_variables(model.variables, rot_rank=0)
               smdp.broadcast_variables(opt.variables, rot_rank=0)
            # Change all_reduce call
            train_loss_value = sndp.oob_allreduce(loss_value) # average the loss across workers
            train_loss(train_loss_value)
            train_accuracy(labels, train_pred)
            return



  Storage Options
    1 S3
       - Moderate and Large datasets
       - satify most Deep Learning use cases
       - file mode: Copy entire dataset to local volume
       - pip mode: Stream dataset from S3

     2. EFS
        - Scalable shared file system
        - no downloading or streaming
        - share file system with other services
        - mount EFS to training cluster
     3. FSx for Lustre file system
       - high performance file system
       - optimized for high-performance
       - natively integrated with S3 (backed up to S3)



  Model Hosting and Deployment

    SageMaker deployment Options
      SageMaker Real-time inference
        - for workloads with low latency requires in the order of milliseconds
        - dedicated instance(s) serve inference requests 
      SageMaker Serverless inference
        - for workloads with intermittent or infrequent traffic patterns
        - do not get to choose compute type, but can choose Memorysize (MB), and MaxCurrency (1 to 200)
        - there is a cold start period after period with no inference requests
      SageMaker Asynchronous inference
        - for inferences with large payload sizes or requiring long processing times
        - can request a queue and process requests asynchronously
        - saves costs, but not for real-time inference
      SageMaker batch transform inference
        - run predictions on batches of data
        - offline processing of large batches of data

  SageMaker Deployment
     
     Model         ---->  Endpoint Config        -----> Endpoint
     CreateModel()         CreateEndpointConfig()       CreateEndpoint()

     Serving               Serverless  or
     container             real-time

     CreateModel()
        - API to register model
        - specify type of serving container (e.g. if TensorFlow, then using TensorFlow serving container)

      CreateEndpointConfig()
         - specify whether to use Serverless endpoint or Real-time endpoint with instance type

     Can switch from Serverless to real-time endpoint
        - by creating a new endpoint
        - call client.update_endpoint()


  resources:
     Blog posts
       medium.com/@shashankprasanna

     Scripts / code samples:
       github.com/shashankprasanna/deeplearning-on-aws

-----------------------------------------------------------
