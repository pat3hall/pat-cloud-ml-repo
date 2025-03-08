------------------------------------------------------
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


    ------------------------------------------------------
Solution
Log in to the Lab Environment

    To avoid issues with the lab, open a new Incognito or Private browser window to log in to the lab. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region.

Create a Step Functions State Machine

    In the search bar on top, type "Step Functions" to search for the Amazon Kinesis service.
    Click on the Step Functions result to go directly to the Step Functions service.
    Click on the State machines button in the sidebar.
    Click on the Create state machine button to launch the wizard to create the state machine.
    When prompted to select a template, opt for a blank state machine.
    Navigate to the Config section of the state machine.
    Update the Execution role to the pre-created step-function-execution-role, which will provide the necessary permissions for the lab.

    Navigate to the Design section of the state machine.
Setup Amazon Transcribe

    Add the transcribe:StartTranscriptionJob action to the start of the workflow.

    Navigate to Amazon S3 in a new tab.

    Open the bucket named stepfunctionsbucket.

    Follow the following path to the GitHub Repository for this lab, and download the conversation.mp3 file:
        Lab GitHub Repository

    Upload conversation.mp3 to the S3 bucket.

    Select the uploaded mp3 file to display the Properties page.

    Copy the S3 URI for the uploaded file.
       s3://cfst-4234-8119d311182a00943842-stepfunctionsbucket-3spgmpuxapru/conversation.mp3

    Return to the Step Functions tab.

     StartTranscriptionJob API doc:
        https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html

    Copy the following details to the API Parameters (replacing the necessary placeholders):

    {
        "LanguageCode": "en-US",
        "Media": {
            "MediaFileUri": "<MP3-File-S3-URI>"
        },
        "TranscriptionJobName": "MyData",
        "OutputBucketName": "<S3-Bucket-Name>"
    }

    Rename the state name to Start Call Transcription Job.

    Add the transcribe:StartTranscriptionJob action after the Start Call Transcription Job state.

    Rename the state name to Check Transcription Job.

    Add a Choice flow state after the Check Transcription Job state.

    Set the conditions for the first Choice state rule to the following condition:
        Variable: $.TranscriptionJob.TranscriptionJobStatus
        Operator: is equal to
        Value: String constant
        COMPLETED

    Add the comment Transcript Completed to the first rule.

    Create a second rule for the Choice state.

    Set the conditions for the first Choice state rule to the following condition:
        Variable: $.TranscriptionJob.TranscriptionJobStatus
        Operator: is equal to
        Value: String constant
        IN_PROGRESS

    Add the comment Transcript Processing to the first rule.

    Add a Wait flow state between the Start Call Transcription Job state, and the Check Transcription Job state.

    Set the wait timer to a fixed interval of 2 seconds.

    Rename the state name to Wait 2 seconds.

    Return to the Choice state.

    Update the next state for the second rule (In Progress) to Wait 2 seconds to create a loop.

    Rename the state name for the Choice state to Transcript Status?.

    Add a Success flow state to the Transcript Completed branch of the Transcript Status? state.

    Add a Fail flow state to the Default branch of the Transcript Status? state, in case of a failure.

    Click on the Create button at the top-right of the screen to save the state machine.

    Click on the Start execution button to trigger the first execution for our state machine.

    Leave all of the contents of the execution window to the defaults, and click Start execution.

    Wait for the job to be completed, and verify that it reaches the Success state.

    Scroll to the top of the page, and click the New execution button to run the state machine again

    Note the error due to the conflicting transcription job name, noting the requirements for a unique name.

    Scroll to the top of the page, and click the Edit state machine button to return to the Workflow Studio.

    Select the Start Call Transcription Job state, and update the API Parameters to the following (replacing the placeholders):

    {
        "LanguageCode": "en-us",
        "Media": {
            "MediaFileUri": "<MP3-File-S3-URI>"
        },
        "TranscriptionJobName.$": "$$.Execution.Name",
        "OutputBucketName": "<S3-Bucket-Name>"
    }

    Select the Check Transcription Job state, and update the API Parameters to the following:

    {
        "TranscriptionJobName.$": "$$.Execution.Name"
    }

    Click on the Save button at the top-right of the screen to save the state machine.

    Click on the Start execution button to trigger the first execution for our state machine.

    Leave all of the contents of the execution window to the defaults, and click Start execution.

    Wait for the job to be completed, and verify that it reaches the Success state.

Setup Amazon Comprehend

    Add the comprehend:DetectSentiment action in between the Transcript Status? state, and the Success state.

    Add the s3:GetObject action in between the Transcript Status? state, and the DetectSentiment state.

    Update the GetObject API Parameters to the following (replacing the placeholders):

    {
        "Bucket": "<S3-Bucket-Name>",
        "Key.$": "States.Format('{}.json', $.TranscriptionJob.TranscriptionJobName)"
    }

    Add a Pass flow state between the GetObject state, and the DetectSentiment state.

    Configure the Parameters of the Pass state, and provide the following definition:

    {
        "Body.$": "States.StringToJson($.Body)"
    }

    Select the DetectSentiment state, and update the API Parameters to the following:

    {
        "LanguageCode": "en",
        "Text.$": "$.Body.results.transcripts[0].transcript"
    }

    Click on the Save button at the top-right of the screen to save the state machine.

    Click on the Start execution button to trigger the first execution for our state machine.

    Leave all of the contents of the execution window to the defaults, and click Start execution.

    Wait for the job to be completed, and verify that it reaches the Success state.

    Return to the Step Functions tab.

    Rename the state name for the GetObject state to Get Transcript File Contents.

    Rename the state name for the Pass state to Parse the JSON.

    Rename the state name for the DetectSentiment state to Detect Sentiment.

    Add a Choice flow state between the Detect Sentiment state, and the Success state.

    Set the conditions for the first Choice state rule to the following condition:
        Variable: $.Sentiment
        Operator: is equal to
        Value: String constant
        NEGATIVE

    Create a second rule for the Choice state.

    Set the conditions for the first Choice state rule to the following condition:
        Variable: $.Sentiment
        Operator: is equal to
        Value: String constant
        POSITIVE

    Add the lambda:Invoke action after the Choice state for the NEGATIVE sentiment path.

    Configure the Function Name for the action to the NegativeInteraction:$LATEST function.

    Set the Next State to Success.

    Add the sqs:SendMessage action after the Choice state for the POSITIVE sentiment path.

    Set the Next State to Success.

    Configure the Queue URL for the action to the PositiveInteractionQueue.

    Add a Pass state to the Default path for the Choice state.

    Click on the Save button at the top-right of the screen to save the state machine.

    Click on the Start execution button to trigger the first execution for our state machine.

    Leave all of the contents of the execution window to the defaults, and click Start execution.

    Wait for the job to be completed, and verify that it reaches the Success state.

Conclusion

Congratulations — you just learned how to create an AWS Step Function to process data and orchestrate events!

    ------------------------------------------------------

------------------------------------------------------
