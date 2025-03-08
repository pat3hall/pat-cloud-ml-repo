
These youtube videos are from Cumulus Cycles by Rob ????

#----------------------------------------------------------------------
Video 1:
Amazon Lex: Build a ChatBot  (BurgerBuddy video 1) by Cumulus Cycles
https://www.youtube.com/watch?v=SmfMQnkx_z8&list=PLRBkbp6t5gM00QA8CMkYIQt7ATE3iJkdN&index=2


#----------------------------------------------------------------------
Video 2:
Amazon Lex: Validate Slot data with Lambda (BurgerBuddy video 2)
  https://www.youtube.com/watch?v=1xRl8Ipa018&list=PLRBkbp6t5gM00QA8CMkYIQt7ATE3iJkdN&index=3

Github:
   https://github.com/CumulusCycles/Amazon_Lex_demo


AWS Lex V2 Using Lamba Function "Interpreting the input event format"

  https://docs.aws.amazon.com/lexv2/latest/dg/lambda-input-format.html

  Lex Event message fields:

    sessionState:
     - current state of the conversation between the user and the Bot

    intent field
      - contains current confirmationState and current slot values

    "intent": {
        "confirmationState": "Confirmed | Denied | None",
        "name": string,
        "slots": {
            // see Slots for details about the structure
        },
        "state": "Failed | Fulfilled | FulfillmentInProgress | InProgress | ReadyForFulfillment | Waiting",
        "kendraResponse": {
            // Only present when intent is KendraSearchIntent. For details, see
    // https://docs.aws.amazon.com/kendra/latest/dg/API_Query.html#API_Query_ResponseSyntax       }
    }
    dialogActintention sub-block
      - indicates the next action Lex should take
        Delegate
          - when Lambda delegates the action to Lex
          - tells Lex, everything looks good so take the next step in the conversation
        ElicitSlot
          - tells Lex which slot should be processed for additional user data

    "dialogAction": {
        "slotElicitationStyle": "Default | SpellByLetter | SpellByWord",
        "slotToElicit": string,
        "type": "Close | ConfirmIntent | Delegate | ElicitIntent | ElicitSlot"
    },

    invocationSource field;
      - The code hook that called the Lambda function. The following values are possible:
         DialogCodeHook
           – Amazon Lex V2 called the Lambda function after input from the user.
         FulfillmentCodeHook
           – Amazon Lex V2 called the Lambda function after filling all the required slots and the intent is ready for fulfillment.

     "invocationSource": "DialogCodeHook | FulfillmentCodeHook",



  Code: BurgerBuddy lambda function code
     # from Amazon_Lex_demo-main/Lambda/lambda_handler.py

# ++++++++++++++++++++++++++++++++++++++++++++

import json

burger_sizes = ['single', 'double', 'triple']
burger_franchises = ['best burger', 'burger palace', 'flaming burger']
best_burger_types = ['plain', 'cheese', 'bacon']
burger_palace_types = ['fried egg', 'fried pickle', 'fried green tomatoes']
flaming_burger_types = ['chili', 'jalapeno', 'peppercorn']


def validate_order(slots):
    # Validate BurgerSize
    if not slots['BurgerSize']:
        print('Validating BurgerSize Slot')

        return {
            'isValid': False,
            'invalidSlot': 'BurgerSize'
        }

    if slots['BurgerSize']['value']['originalValue'].lower() not in burger_sizes:
        print('Invalid BurgerSize')

        return {
            'isValid': False,
            'invalidSlot': 'BurgerSize',
            'message': 'Please select a {} burger size.'.format(", ".join(burger_sizes))
        }

    # Validate BurgerFranchise
    if not slots['BurgerFranchise']:
        print('Validating BurgerFranchise Slot')

        return {
            'isValid': False,
            'invalidSlot': 'BurgerFranchise'
        }

    if slots['BurgerFranchise']['value']['originalValue'].lower() not in burger_franchises:
        print('Invalid BurgerSize')

        return {
            'isValid': False,
            'invalidSlot': 'BurgerFranchise',
            'message': 'Please select from {} burger franchises.'.format(", ".join(burger_franchises))
        }

    # Validate BurgerType
    if not slots['BurgerType']:
        print('Validating BurgerType Slot')

        return {
            'isValid': False,
            'invalidSlot': 'BurgerType'
        }

    # Validate BurgerType for BurgerFranchise
    if slots['BurgerFranchise']['value']['originalValue'].lower() == 'best burger':
        if slots['BurgerType']['value']['originalValue'].lower() not in best_burger_types:
            print('Invalid BurgerType for Best Burger')

            return {
                'isValid': False,
                'invalidSlot': 'BurgerType',
                'message': 'Please select a Best Burger type of {}.'.format(", ".join(best_burger_types))
            }

    if slots['BurgerFranchise']['value']['originalValue'].lower() == 'burger palace':
        if slots['BurgerType']['value']['originalValue'].lower() not in burger_palace_types:
            print('Invalid BurgerType for Burger Palace')

            return {
                'isValid': False,
                'invalidSlot': 'BurgerType',
                'message': 'Please select a Burger Palce type of {}.'.format(", ".join(burger_palace_types))
            }

    if slots['BurgerFranchise']['value']['originalValue'].lower() == 'flaming burger':
        if slots['BurgerType']['value']['originalValue'].lower() not in flaming_burger_types:
            print('Invalid BurgerType for Flaming Burger')

            return {
                'isValid': False,
                'invalidSlot': 'BurgerType',
                'message': 'Please select a Flaming Burger type of {}.'.format(", ".join(flaming_burger_types))
            }

    # Valid Order
    return {'isValid': True}


def lambda_handler(event, context):
    print(event)

    bot = event['bot']['name']
    slots = event['sessionState']['intent']['slots']
    intent = event['sessionState']['intent']['name']

    order_validation_result = validate_order(slots)

    # DialogCodeHook - lambda was called when Lex was processing user data
    if event['invocationSource'] == 'DialogCodeHook':
        # if invalid order value, set dialogAction to SlotToElicit for invalid slot
        #   (include a message if order_validation_result returned a message)
        if not order_validation_result['isValid']:
            if 'message' in order_validation_result:
                response = {
                    "sessionState": {
                        "dialogAction": {
                            "slotToElicit": order_validation_result['invalidSlot'],
                            "type": "ElicitSlot"
                        },
                        "intent": {
                            "name": intent,
                            "slots": slots
                        }
                    },
                    "messages": [
                        {
                            "contentType": "PlainText",
                            "content": order_validation_result['message']
                        }
                    ]
                }
            else:
                response = {
                    "sessionState": {
                        "dialogAction": {
                            "slotToElicit": order_validation_result['invalidSlot'],
                            "type": "ElicitSlot"
                        },
                        "intent": {
                            "name": intent,
                            "slots": slots
                        }
                    }
                }
        else:
            response = {
                "sessionState": {
                    "dialogAction": {
                        "type": "Delegate"
                    },
                    "intent": {
                        'name': intent,
                        'slots': slots
                    }
                }
            }

    # FulfullmentCodeHook - lambda was called by Lex during intent fulfillment step
    #  set dialogAction state to 'Close' which will close the conversation flow in lex
    if event['invocationSource'] == 'FulfillmentCodeHook':
        response = {
            "sessionState": {
                "dialogAction": {
                    "type": "Close"
                },
                "intent": {
                    "name": intent,
                    "slots": slots,
                    "state": "Fulfilled"
                }

            },
            "messages": [
                {
                    "contentType": "PlainText",
                    "content": "I've placed your order."
                }
            ]
        }

    print(response)
    return response

# ++++++++++++++++++++++++++++++++++++++++++++



#----------------------------------------------------------------------
Video 3:

Amazon Lex: Integrate ChatBot into a Web Page
  https://www.youtube.com/watch?v=cI1NAjLE_I8&list=PLRBkbp6t5gM00QA8CMkYIQt7ATE3iJkdN&index=4

Github (same as for video 2):
   https://github.com/CumulusCycles/Amazon_Lex_demo

  Video overview:
    integrate our BurgerBuddy ChatBot into a Web Page, hosted in S3, using Kommunicate.

  Kommunicate
    - a service that allows you to integrate chatbots into a webpage
    - need a account (can use free 30 day account)


1. Setup - integrate Lex with Kommunicate

  AWS Console -> IAM -> User Group -> Create User Group ->
       Name: LexAdmin, select user: <LexUser> Policy: AmazonLexFullAccess -> Create Group


  AWS Console -> IAM -> User -> select <LexUser> -> security -> create access key
     -> Copy/save access key / secret access key and provide it to Kommunicate


    Kommunicate setup
      - create a 30 day free account
      - go to Bot integration dashboard by selecting Bot Integration Icon in left navigation menu, then select "Amazon Lex"
         - select LexV2
         - add AWS IAM access key and secret access key
         - select Region: Us-east-1, select Bot Name in Lex platorm: BurgerBuddy, select Bot Alias: TestBotAlias,
           default language: English(US)
           -> Save and proceed
           -> Setup your Bot:
               BotName: BurgerBuddy
               Bot Photo: <useDefaultPhoto>
               -> Save and Proceed
           -> Automatic bot to human handoff:
             select "Disable this feature"
               -> Finish bot setup
           -> select "Try out your new bot BurgerBuddy"
              -> Demo Conversation ->
                 try conversation
                   ex: I like to order a burger; triple; best burger; bacon; yes


2. Setup - integrate Kommunicate with S3 Webpage
    for webpage use: burgerbuddy_chatbot_tutorial\Amazon_Lex_demo-main\WebPage\index.html

    Note: in Demo, he initially hosts the burgerbuddy on a local webserver (e.g. 127.0.0.1/Webhost)


    Kommunicate setup
       -> click on setup icon <gear> -> install <left tab>
           -> copy provided Java Script
           -> in provided "index.html" file, paste Kommicate Java Script code in bottom of <head> section

    AWS -> S3 -> Create S3 burgerbuddybot-ph bucket, uncheck "block all public access, acknowledge -> Create bucket
           -> under bucket properties -> Static website hosting -> edit ->
                "enable" static website hosting, Index Document: index.html, Error Document: error.html -> Save changes
           -> under bucket permissions -> Edit Bucket Policy -> add below policy -> save changes

           S3 bucket policy:

               {
               "Version": "2012-10-17",
               "Statement": [{
                 "Sid": "PublicReadGetObject",
                 "Effect": "Allow",
                 "Principal": "*",
                 "Action": [
                      "s3:GetObject"
                      ],
                 "Resource": [
                     "arn:aws:s3:::burgerbuddybot-ph/*"
                  ]
                 }
               }]

         -> bucket -> upload "index.html" and "img" subdirectory

         -> click on "index.html" object -> copy "Object html" -> go to this html page

#----------------------------------------------------------------------

  Video 4:

    Amazon Lex: Add Response Cards
      https://www.youtube.com/watch?v=grBCFLscjQ0&list=PLRBkbp6t5gM00QA8CMkYIQt7ATE3iJkdN&index=5

   Video overview:
      In this video, we'll add Response Cards to our Bot.
         -> provides a more user friend response click on response options

Github (same as for video 2 & 3):
   https://github.com/CumulusCycles/Amazon_Lex_demo


#----------------------------------------------------------------------

 Video 5:
    Amazon Lex - Slack Integration

      https://www.youtube.com/watch?v=fak-223hHTE&list=PLRBkbp6t5gM00QA8CMkYIQt7ATE3iJkdN&index=6

      In this video, we'll integrate Amazon Lex with Slack.

        -> Slack integration will be via Lex Deploy -> Channel

        -> Lex support 3 channel types: Facebook, Slack, and Twilio SMS

