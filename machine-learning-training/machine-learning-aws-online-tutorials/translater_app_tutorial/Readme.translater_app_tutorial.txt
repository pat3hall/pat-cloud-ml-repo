This directory contains:

Build A Translator App in 30 Min or Less
  https://community.aws/content/2drbcpmaBORz25L3e74AM6EIcFj/build-your-own-translator-app-in-less-30-min
  Use Amazon Translate, Amazon Comprehend, Amazon Lambda, Amazon Polly, and Amazon Lex to bring a 
  translation application to life and test it in 30 minutes or less.


Solution Overview
  In this tutorial you are going to create a translator chatbot app, with Amazon Lex that will handle the 
  frontend interaction with the user, and the backend will be in charge of an AWS Lambda Function with 
  the AWS SDK for Python library Boto3 code using the following AWS services:

  - Amazon Comprehend in charge of detecting the language entered.
  - Amazon Translate it will translate into the desired language.
  - Amazon Polly it delivers the audio with the correct pronunciation.


Build It in Seven Parts

    Part 1 - Create the Function That Detects the Language and Translates It Into the Desired Languag 🌎.
    Part 2 - Create the Function to Converts Text Into Lifelike Speech 🦜.
    Part 3 - Configure the Chatbot Interface With Amazon Lex🤖.
    Part 4 - Build the Interface Between the Backend and the Frontend.
    Part 5 - Integrate the Backend With the Frontend.
    Part 6 - Let’s Get It to Work!
    Part 7 - Deploy Your Translator App.


Create S3 bucket
  -> bucket: translator-app-bkt


Create Lambda function: translatorAppFcn

 AWS Console -> Lambda -> Create Function -> Author from Scratch ->
   Name: translatorAppFcn, Runtime: Python 3.12, Create New Role with Lambda permissions,
     Create a new role with basic Lambda permssions
     -> Create Function

add inline policy to Lambda role, translatorFcn-role-qhsjbf6:

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "VisualEditor0",
                "Effect": "Allow",
                "Action": [
                    "polly:SynthesizeSpeech",
                    "polly:StartSpeechSynthesisTask",
                                    "polly:GetSpeechSynthesisTask",
                    "comprehend:DetectDominantLanguage",
                    "translate:TranslateText"
                ],
                "Resource": "*"
            },
            {
                "Sid": "VisualEditor1",
                "Effect": "Allow",
                "Action": [
                    "s3:PutObject",
                    "s3:GetObject"
                ],
                "Resource": [
                    "arn:aws:s3:::translator-app-bkt/*",
                    "arn:aws:s3:::translator-app-bkt"
                ]
            }
        ]
    }



translatorAppFcn lambda code:
   +++++++++++++++

import boto3

# Functions to calls TranslateText and perform the translation:

translate_client = boto3.client('translate')

def TranslateText (text,sourceLanguage, tragetLanguage):
    response = translate_client.translate_text(
    Text=text,
    SourceLanguageCode=sourceLanguage,
    TargetLanguageCode=tragetLanguage  
    )
    text_ready =  response['TranslatedText'] 
    return text_ready


def get_target_voice(language):
    to_polly_voice = dict( [ ('en', 'Amy'), ('es', 'Conchita'), ('fr', 'Chantal'), ('pt-PT', 'Cristiano'),('it', 'Giorgio'),("sr","Carmen"),("zh","Hiujin") ] )
    target_voice = to_polly_voice[language]
    return target_voice


import boto3
polly_client = boto3.client('polly')
def start_taskID(target_voice,bucket_name,text):
    print(f'calling polly; target_voice: {target_voice}, bucket_name: {bucket_name}, text: {text}')
    response = polly_client.start_speech_synthesis_task(
                    VoiceId=target_voice,
                    OutputS3BucketName = bucket_name,
                    OutputFormat= "mp3", 
                    Text= text,
                    Engine= "standard")

    status = response['SynthesisTask']['TaskStatus']
    task_id = response['SynthesisTask']['TaskId']
    object_name = response['SynthesisTask']['OutputUri'].split("/")[-1] 
    print(f'start_taskID returning: task_id: {task_id}, object_name: {object_name}, status: {status}')
    return task_id, object_name, status


import time
def get_speech_synthesis(task_id):
    # return current epoch time in sec + 3 
    max_time = time.time() + 3 
    print(f'called get_speech_synthesis: task_id: {task_id}')
    while time.time() < max_time:
        response_task = polly_client.get_speech_synthesis_task(
        TaskId=task_id
        )
        print(f'polly response_task: {response_task}')
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
        time.sleep(5/1000)
    return status


s3_client = boto3.client("s3")

def create_presigned_url(bucket_name, object_name, expiration=3600):
    value = object_name.split("/")[-1]
    response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name, 'Key': value},
                                                    ExpiresIn=expiration)
    return response

def get_intent(lex_event):
    interpretations = lex_event['interpretations'];
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


# note: I modified to work with Lex V2
def lambda_handler(event, context):
    print(event)
    bucket_name = "translator-app-bkt"
    #Lambda Function Input Event and Response Format
    interpretations = event['interpretations']
    intent_name = interpretations[0]['intent']['name']
    intent = get_intent(event)


    slots = event['sessionState']['intent']['slots']
    # intent = event['sessionState']['intent']['name']

    #need it to Response Format
    active_contexts = get_active_contexts(event) 
    session_attributes = get_session_attributes(event) 
    #to find out when Amazon Lex is asking for text_to_translate and join the conversation.
    previous_slot_to_elicit = session_attributes.get("previous_slot_to_elicit") 
    print(session_attributes)
    
    if intent_name == 'TranslateIntent':
        print(intent_name)
        print(intent)
        to_language = get_slot('to_language',intent)
        from_language = get_slot('from_language',intent)
        text_to_translate = get_slot("text_to_translate",intent, preference = "originalValue")
        print(f'to_language: {to_language}  from_language: {from_language},  text_to_translate: {text_to_translate}')

        # if to or from language not yet specified, request the language
        if to_language == None or from_language == None:

            #if to_language == None and from_language == None:
            #    print ("Neither 'to_language' nor 'from_language' set, Delegate to Lex")
            #    response = {
            #        "sessionState": {
            #            "dialogAction": { "type": "Delegate" },
            #            "intent": { 'name': intent_name, 'slots': slots }
            #        }
            #    }
            #    return response
    
            if from_language == None:
                print ("'from_language' not set, ElicitSlot from_language to Lex")
                message = "What language do you want to translate from?"
                response = {
                        "sessionState": {
                            "dialogAction": { "slotToElicit": "from_language", "type": "ElicitSlot" },
                            "intent": { "name": intent_name, "slots": slots }
                        },
                        "messages": [ { "contentType": "PlainText", "content": message } ]
                    }
                return response
            elif to_language == None:
                print ("'to_language' not set, ElicitSlot to_language to Lex")
                message = "What language do you want to translate to?"
                response = {
                        "sessionState": {
                            "dialogAction": { "slotToElicit": "to_language", "type": "ElicitSlot" },
                            "intent": { "name": intent_name, "slots": slots }
                        },
                        "messages": [ { "contentType": "PlainText", "content": message } ]
                    }
                return response
                

            else:
                print('ERROR: This branch should not occur')
    
            
        # if language specified, but text to translate not provided, request the text
        elif (text_to_translate == None) and (to_language != None) and (from_language != None):
            print("'text_to_translate' not set so ElicitSlot text_to_translate to Lex")
            message = "What text do you want to translate?"
            response = {
                    "sessionState": {
                        "dialogAction": { "slotToElicit": "text_to_translate", "type": "ElicitSlot" },
                        "intent": { "name": intent_name, "slots": slots }
                    },
                    "messages": [ { "contentType": "PlainText", "content": message } ]
                }
            return response

            
        # call Amazon Translate to translate the text, the call polly to convert translated text to voice message, 
        #  return voice message to Lex
        else:
            print("diferente a none")
            text_ready = TranslateText(text_to_translate,from_language,to_language)
            target_voice = get_target_voice(to_language)
            task_id, object_name, polly_status = start_taskID(target_voice,bucket_name,text_ready)
            
            url_short = create_presigned_url(bucket_name, object_name, expiration=3600)
            
            print(f'url_short: {url_short}')
            print ("text_ready: ", text_ready)
            if polly_status != "completed":
                status = get_speech_synthesis(task_id)
            
            message = f"The translate text is: {text_ready}. Hear the pronunciation here {url_short} "
            
            #print(elicit_intent(active_contexts, session_attributes, intent, messages))
            #return elicit_intent(active_contexts, session_attributes, intent, messages)
            print ("return ElicitIntent with translated texted")

            response = {
                    "sessionState": {
                        "dialogAction": { "slotToElicit": "text_to_translate", "type": "ElicitIntent" },
                        "intent": { "name": intent_name, "slots": slots }
                    },
                    "messages": [ { "contentType": "PlainText", "content": message } ]
                }
            return response
   +++++++++++++++


Create Lex translatorAppBot


 AWS console -> Lex -> Create bot ->
    Creation Method: Traditional, Create a blank bot, bot name: translatorAppBot,
    IAM: Create a role with .. lex permissions,
    COPPA: no,
    -> Next -> 
    Language: English (US), voice interaction: Danielle 
    -> Done

    Note: Role created: AWSServiceRoleForLexV2Bots_CPOLQLQ0V0O

    Intent: WelcomeIntent
      slots -> add slot -> Name: text_to_translate, Slot type: AMAZON.AlphaNumber, Prompts: Provide text to be translated
      slots -> add slot -> Name: language, Slot type: languageSlotType, Prompts: Translate to language
         Slot Type: languageSlotType, Restrict to slot values,
            Slot type Values: en: English, english, ingles; es: Spanish, spanish, espanol; fr: French, french; it: Italian, italian, italiano;
      Sample Utterances: I want to translate, I want to do a translate, I want to translate to {language}, translate
      Code Hooks: "Use a Lambda funtion for initialiation and validation"
      

    Sample Utterrance: hi, hello, hey, help, i need help 
    Initial responce -> Message: I can help with that
    Advanced options -> default flow
    Response -> Default flow -> Response -> variation -> advanced option ->
     Slots prompts -> Add -> Card Group -> 
      Title: What would you like to do:
      buttons -> add button ->
      Button 1 title: Order Ice Cream, Value: create order
      Button 2 title: Cancel Order, Value: Cancel Order
      -> Update responses -> 
    -> Save Intent

