------------------------------------
 Amazon Mechanical Turk & Ground Truth
------------------------------------
 FROM A Cloud Guru  AWS ML training courses:

    Amazon Mechanical Turk
    - a  crowdsourcing marketplace that makes it easier for individuals and businesses to outsource their processes and jobs 
      to a distributed workforce who can perform these tasks virtually. 
    - includes anything from conducting simple data validation and research to more subjective tasks like survey participation,
      content moderation, and more. 

      to find
      AWS -> SageMaker -> Ground Truth -> Labeling workforce



 Amazon Sagemaker Ground Truth Service
    - tool that helps build ground truth datasets by allowing different type os tagging/labeling processes
    - easily create labeled data

  SageMaker Ground Truth
    - a labeling services that leverages the human feedback to improve prediction accuracy

    Self Serve Model
                           |------------------------------|
       Input Data (S3) --> |labeling Workforce    Label UI|  --> Output data (S3)
                           |------------------------------|
                               SageMaker Ground Truth     

     Type of Workforce
       Amazon Mechanical Turk
         - vendor-managed workforce
         - anyone can sign up to this service and become an MTurk worker
       Private workforce
         - if you have a highly confidential data, then you can create your own private workforce to 
           review and label the dataset
       AI application
         - you can use any AI applications like Amazon Recognition, Textract, Comprehend and Transcribe.

      Managed Service Model
        Amazon SageMaker Ground Truth Plus
           - branded as a turnkey service that leverages their expert workforce to deliver high quality 
             training dataset, which also costs 40% cheaper.  
           flow:
           - With this service, once the data is uploaded to an S3 bucket,
           - once the data is uploaded to an S3 bucket, Ground Truth Plus will set up a data labeling 
             workflow that meets customer privacy and security requirements and operates on behalf of the customer.
           - These workflows are analyzed either by Amazon employed workforce or a third party vendor who 
             are expertly trained on a variety of data labeling jobs.
           - The customer can monitor the progress, offer feedback and review as the data is getting labeled

                           |------------------------------|
       Input Data (S3) --> |workflow   Workforce   Monitor|  --> Output data (S3)
                           |------------------------------|
                             SageMaker Ground Truth Plus     


------------------------------------
Amazon Mechanical Turk FAQs
  https://www.mturk.com/help


What is MTurk?
  - Amazon Mechanical Turk (MTurk) provides a service for service requesters (hereafter “Requesters”) to integrate Artificial 
    Intelligence directly into their applications by making requests of humans. 
  - Requesters can use the MTurk web user interface or web services API to submit tasks to the MTurk web site, approve completed 
    tasks, and incorporate the answers into their applications. 

What problem does Amazon Mechanical Turk solve? 
  software developers
  -  give developers a programmable interface to a network of humans to solve these kinds of problems and incorporate this 
     human intelligence into their applications
  businesses and entrepreneurs 
   - tasks completed, the MTurk service solves the problem of accessing a vast network of human intelligence with the efficiencies 
     and cost-effectiveness of computers.

Example Request API Code:
  - All Requesters need to do is write normal code. The pseudo code below illustrates how simple this can be.

      read(photo);
      photoContainsHuman = callMechanicalTurk(photo);
    
      if (photoContainsHuman == TRUE) {
          acceptPhoto();
      } else {
          rejectPhoto();
      }


HITs :Human Intelligence Tasks

Who are Amazon Mechanical Turk Masters?
  - Amazon Mechanical Turk (MTurk) has built technology which analyzes Worker performance, identifies high performing 
    Workers, and monitors their performance over time. 
  - Workers who have demonstrated excellence across a wide range of HITs are awarded the Masters Qualification


------------------------------------

 How to set up A Project On Amazon Mechanical Turk (MTurk): An MTurk Tutorial
 https://www.youtube.com/watch?v=1Rv0miGwr2s

  Summary:
    This video shows you the necessary requirements for setting up a project within Amazon Mechanical Turk (MTurk). 
    This video is helpful for anyone needing a tutorial on Amazon Mechanical Turk, especially those conducting online 
    survey research including undergraduate students, graduate students, doctoral students, market/marketing researchers, 
    psychology researchers, consumer behavior researchers, etc.

    -> two part: Part 1: Setup in Amazon Mechanical Turk; Part 2: set in Qualtrics

    to use: 
      mturk.com
        -> Sign as requester (create an account, use AWS account, or Amazon account)
          -> Note: need to create Mturk account associated with above accounts

          # create a new Project

          -> New Project -> Survey -> Survey Link (option for linking to Qualtrics) -> 
            Enter Properties:
              Project Name: <projectName>, 
              Describe Survey to workers: Title, Descripion, keywords
              Setting upt your Survey:
                Rewards per response: <$>, [max] Number of Respondents: <number>, Time allotted per Worker: <time>, Survey Expires in: <days>,
                Auto-approve and pay Workers in: <timeFrame>
              Worker requirements
                Require that workers by Master to do your tasks: <yes/no>, 
                Additional Requirments: <e.g. HIT Approval Rate... greater than 98; Number of HITS approved greater than 50> 
                    Premium options: Age range, Location, Borrower, Blogger, ...
                Project contains adult content: <?>
                Task Visibility:
                  Public: All workers can see and preview my tasks
                  Private: All workers can see my tasks but only workers that meet all qualification requirements can preview my tasks
                  Hidden: ONly workers that meet my Qualification requirements can see and preview my tasks

             Note:
               Project types include: 
                      Survey: Survey, Survey Link, 
                      Vision: Image Classification Bounding Box, Semantic Segmentaiton, ...
                      Language: Sentiment Analysis, Intent Detection, Collect Utterance, Emotion Detection, ...
                      Other: Data Collection, Website Collection, Website Classication
  
            Design Layout:
              Survey Link Instructions:
                 <modify template to meet your needs>
                 Survey Link: <changed to the Qualtrics link ???>

            Preview and Finish
              -> Finish this batch

      select <batch>
         -> Publish -> Review layout, cost summary, add payment method

------------------------------------

SageMaker GroundTruth Hands-On Tutorial: Labeling Images and Text
  ML Workbench
  https://www.youtube.com/watch?v=PpjlM-EdVrc

  Summary
    In this video, we give a quick tutorial using SageMaker GroundTruth to label both images and sensor logs (CSV). 
    In this tutorial we will:
       1. Upload data
       2. Create a private workgroup
       3. Label a single-class dataset
       4. Label a multi-class dataset

  SageMaker Ground Truth
    - tool for manually and automatically labeling datasets

   Example Dataset
     - using two data modalities
        1. images from laptop background
        2. CSV around Network logs - malicious activity use case

 step 1: Upload data to S3 bucket
     image folder (Wallpapers) 
        -> upload desktop images (148)


  Workforce Labeling options:
    Amazon Mechanical Turk
      - A team of global, on-demand workers powered by Amazon Mechanical Turk.
      - users of mechanical turk (mturk.com) label your dataset
    Private
      - A team of workers from your organization.
    Vendor
      - A selection of experienced vendors who specialize in providing data labeling services.
      - 3rd party service labeling service
      - sends you to AWS Marketplace labeling services
    
  step 2: Create Labeling Workforce
    AWS -> SageMaker -> Ground Truth -> Labeling Workforce ->
    # demo uses private workforce
     -> Private <top center tab> -> Create private team -> 
       Private Team Creation: Create a private team with AWS Cognito
       Team Name: data-quality-team
       Add workers: Invite new workers by email
       email addresses:
       Organization Name
       Contact email:
       -> Create Private Team


    # where to manage your private team
    AWS -> SageMaker -> Ground Truth -> Labeling Workforce -> Private -> data-quality-team

  step 2: Create Labeling Job

    AWS -> SageMaker -> Ground Truth -> Labeling Job -> Create labeling Job ->
      Job name: bg-labeling
      Input Data Setup: Automated data setup
      Data setup: S3 location: s3://pat-demo-bkt/background-images/, Output dataset: Same location as input dataset,
      Data Type: Image
      IAM Role: Create Role: Specific S3 bucket: pat-demo-bkt -> Create
      -> Complete Data Setup
      Task selection: : Imagage Classification (Multi-label)
      Worker Types: Private
      Private Teams: data-quality-team
      task timemout: 5 min, task expiration time: 2 days
      Image classification (Multi-label) labeling tool: <add labels>
      -> Create

      Note: 
        Option for "Enable Automated data label"
          - SageMaker will automatically label a portion of your dataset. It will train a model in your AWS 
            account using Built-in Algorithm and your dataset. When you enable this, training jobs use new computing
            resources on your behalf. Pricing info link
          - need 1000 or more datapoints, will hold out a percentage for validation, then ask you to verify 
            if the the validation set was correctly labeled

  step 3: Label dataset
     option a: use email link sent with temp password
     option b: 
      AWS -> SageMaker -> Ground Truth -> Labeling Workforce -> Private -> Labeling portal sign-in URL ->

      -> Start working
      label images

  step 3: Examine Label dataset in S3 bucket

      AWS S3 -> bucket/folder -> pat-demo-bkt/background-images -> "bg-label" subfolder

        labeling created subfolders:

         annotation-tool/
         annotations/
            annotations/consolidated-annotation/consolidation-response/iteration-1/*.json
               -> json file for each image with labeling info 
                  example:
                  [{"datasetObjectId":"5","consolidatedAnnotation":
                      {"content":{"bg-labeling":[6,0],"bg-labeling-metadata":
                        {"job-name":"labeling-job/bg-labeling","confidence-map":
                           {"6":0.0,"0":0.0},"class-map":{"6":"clouds","0":"Canyon"},
                            "type":"groundtruth/image-classification-multilabel",
                            "human-annotated":"yes","
                            creation-date":"2024-09-27T21:31:32.051349"
                         }}
                      }}]

         manifests/
         temp/
	

   clean up
      -> delete private group
      -> emtpy bucket
------------------------------------


