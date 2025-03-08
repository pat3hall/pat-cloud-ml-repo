import boto3

Functions to calls TranslateText and perform the translation:

translate_client = boto3.client('translate')

def TranslateText (text,language):
    response = translate_client.translate_text(
    Text=text,
    SourceLanguageCode="auto",
    TargetLanguageCode=language  
    )
    text_ready =  response['TranslatedText'] 
    return text_ready


Translate / Client / translate_text doc:
  https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/translate/client/translate_text.html

  Translate.Client.translate_text(**kwargs)
    - Translates input text from the source language to the target language.
    Required Parameters:
      Text (string) – 
        - The text to translate
      SourceLanguageCode (string) –
        - The language code for the language of the source text.
        - If you specify 'auto', Amazon Translate will call Amazon Comprehend to determine the source language.
      TargetLanguageCode (string) –
        - The language code requested for the language of the target text.


#Match the language code from Amazon Translate with the right voice from Amazon Polly.

def get_target_voice(language):
    to_polly_voice = dict( [ ('en', 'Amy'), ('es', 'Conchita'), ('fr', 'Chantal'), ('pt-PT', 'Cristiano'),('it', 'Giorgio'),("sr","Carmen"),("zh","Hiujin") ] )
    target_voice = to_polly_voice[language]
    return target_voice


Amazon Translate supported voices
  https://docs.aws.amazon.com/translate/latest/dg/what-is-languages.html

polly available voices:
  https://docs.aws.amazon.com/polly/latest/dg/available-voices.html


Functions to Calls the Amazon Lex APIs

a. StartSpeechSynthesisTask:

import boto3
polly_client = boto3.client('polly')
def start_taskID(target_voice,bucket_name,text):
    response = polly_client.start_speech_synthesis_task(
                    VoiceId=target_voice,
                    OutputS3BucketName = bucket_name,
                    OutputFormat= "mp3", 
                    Text= text,
                    Engine= "standard")

    task_id = response['SynthesisTask']['TaskId']
    object_name = response['SynthesisTask']['OutputUri'].split("/")[-1] 
    return task_id, object_name


b. GetSpeechSynthesisTask:

import time
def get_speech_synthesis(task_id):
    # return current epoch time in sec + 2 
    max_time = time.time() + 2 
    while time.time() < max_time:
        response_task = polly_client.get_speech_synthesis_task(
        TaskId=task_id
        )
        status = response_task['SynthesisTask']['TaskStatus']
        print("Polly SynthesisTask: {}".format(status))
        if status == "completed" or status == "failed":
            if status == "failed": 
                TaskStatusReason = response_task['SynthesisTask']['TaskStatusReason']
                print("TaskStatusReason: {}".format(TaskStatusReason))
            else:
                value= response_task['SynthesisTask']['OutputUri']
                print("OutputUri: {}".format(value))
            break
            
    time.sleep(2)
    return status


Polly / Client / start_speech_synthesis_task doc
  https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/polly/client/start_speech_synthesis_task.html

  start_speech_synthesis_task
    Polly.Client.start_speech_synthesis_task(**kwargs)
    - Allows the creation of an asynchronous synthesis task, by starting a new SpeechSynthesisTask.

Polly / Client / get_speech_synthesis_task
  https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/polly/client/get_speech_synthesis_task.html
  get_speech_synthesis_task

  Polly.Client.get_speech_synthesis_task(**kwargs)
    - Retrieves a specific SpeechSynthesisTask object based on its TaskID. This object contains information about the given 
    speech synthesis task, including the status of the task, and a link to the S3 bucket containing the output of the task. 



Generate Presigned URL to Access the [S3] Audio File From Anywhere


s3_client = boto3.client("s3")

def create_presigned_url(bucket_name, object_name, expiration=3600):
    value = object_name.split("/")[-1]
    response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': value},
                                                    ExpiresIn=expiration)
    return response



S3 / Client / generate_presigned_url
   generate_presigned_url
   https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/generate_presigned_url.html

   S3.Client.generate_presigned_url(ClientMethod, Params=None, ExpiresIn=3600, HttpMethod=None)
     - Generate a presigned url given a client, its method, and arguments

   Parameters:
        ClientMethod (string) – The client method to presign for

        Params (dict) – The parameters normally passed to ClientMethod.

        ExpiresIn (int) – The number of seconds the presigned url is valid for. By default it expires in an hour (3600 seconds)




Create Lex Bot:

   AWS Console -> Lex -> Create Bot ->
      Creation Method: Traditional, "Create A blank bot", Bot Name: translateAppBot, 
        IAM Permissions: Create a Role with Lex permissions
          Note: Role Name: AWSServiceRoleForLexV2Bots_E6KZVPFAOH,
          CoPPA: No, 
          -> Next -> 
          Language: English  -> Done

Functions used by Lambda:


def get_intent(intent_request):
    interpretations = intent_request['interpretations'];
    if len(interpretations) > 0:
        return interpretations[0]['intent']
    else:
        return None;

def get_slot(slotname, intent, **kwargs):
    try:
        slot = intent['slots'].get(slotname)
        if not slot:
            return None
        slotvalue = slot.get('value')
        if slotvalue:
            interpretedValue = slotvalue.get('interpretedValue')
            originalValue = slotvalue.get('originalValue')
            if kwargs.get('preference') == 'interpretedValue':
                return interpretedValue
            elif kwargs.get('preference') == 'originalValue':
                return originalValue
            # where there is no preference
            elif interpretedValue:
                return interpretedValue
            else:
                return originalValue
        else:
            return None
    except:
        return None

def get_active_contexts(event):
    try:
        return event['sessionState'].get('activeContexts')
    except:
        return []

def get_session_attributes(event):
    try:
        return event['sessionState']['sessionAttributes']
    except:
        return {}

Lambda Handler Code:

def lambda_handler(event, context):
    print(event)
    #Lambda Function Input Event and Response Format
    interpretations = event['interpretations']
    intent_name = interpretations[0]['intent']['name']
    intent = get_intent(event)
    #need it to Response Format
    active_contexts = get_active_contexts(event) 
    session_attributes = get_session_attributes(event) 
    previous_slot_to_elicit = session_attributes.get("previous_slot_to_elicit") #to find out when Amazon Lex is asking for text_to_translate and join the conversation.
    print(session_attributes)
    
    if intent_name == 'TranslateIntent':
        print(intent_name)
        print(intent)
        language = get_slot('language',intent)
        text_to_translate = get_slot("text_to_translate",intent)
        print(language,text_to_translate)

        # if language not yet specified, request the language
        if language == None:
            print(language,text_to_translate)
            return delegate(active_contexts, session_attributes, intent)
            
        # if language specified, but text to translate not provided, request the text
        if (text_to_translate == None) and (language != None) and (previous_slot_to_elicit != "text_to_translate"):
            print(language,text_to_translate)
            response = "What text do you want to translate?"
            messages =  [{'contentType': 'PlainText', 'content': response}]
            print(elicit_slot("text_to_translate", active_contexts, session_attributes, intent, messages))
            return elicit_slot("text_to_translate", active_contexts, session_attributes, intent, messages)
            
        # call Amazon Translate to translate the text, the call polly to convert translated text to voice message, 
        #  return voice message to Lex
        if previous_slot_to_elicit == "text_to_translate": 
            print("diferente a none")
            text_to_translate = event["inputTranscript"]
            text_ready = TranslateText(text_to_translate,language)
            target_voice = get_target_voice(language)
            object_name,task_id = start_taskID(target_voice,bucket_name,text_ready)
            
            url_short = create_presigned_url(bucket_name, object_name, expiration=3600)
            
            print ("text_ready: ", text_ready)
            status = get_speech_synthesis(task_id)
            
            response = f"The translate text is: {text_ready}. Hear the pronunciation here {url_short} "
            messages =  [{'contentType': 'PlainText', 'content': response}]
            
            print(elicit_intent(active_contexts, session_attributes, intent, messages))
            return elicit_intent(active_contexts, session_attributes, intent, messages)
