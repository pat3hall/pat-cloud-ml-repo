------------------------------------------------------
9.10 Demo: Deconstruction
   -> Clean-up UFO related labs


   AWS console -> API Gateway ->  select "ufo-inference-api -> Delete

   AWS console -> SageMaker -> Inference -> Endpoints ->   select "linear-learner*" -> Action -> Delete
                                         -> Endpoint Configuration -> select "linear-learner*" -> Action Delete

   AWS console -> SageMaker -> Notebooks -> select "my-notebook-inst -> Action -> Stop
                                                                     -> Action -> Delete

   AWS Console -> S3 -> select bucket -> Empty
                     -> select bucket -> Delete

   AWS Console -> Lambda -> select function: invoke-sagemaker-endpoint -> Action -> Delete

------------------------------------------------------

