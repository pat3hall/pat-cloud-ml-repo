1:17 / 40:14
Amazon Lex Chatbot Tutorial - Introduction and Demo - with AWS Lambda | Lex V2 Architecture Example 
https://www.youtube.com/watch?v=Am0_8MKgft8


Lex Key Concepts

  Intent
    - a goal that users want to achieve
    - e.g. order an ice cream / cancel an order
    FallbackIntent
     - built-in intent
     - when unexpected user input, fallbackIntent is invoked by default 
     - does NOT included Utterances, slots, and confirmation prompt options
  Utterance
    - anything that users says
    - e.g. I want ot place an order / Can you cancel that order
  Slots
    - values provided by user to fulfill an intent
    - e.g. Chocolate (flavor), Large (size)
  Confirmation
    - confirm with user before proceeding with fulfullment
    - e.g We are now ready to place an order for Large Chocolate Ice Cream. 
      Please confirm (yes/No)
  Fulfilment
    - fulfiling the Intent
    - E.g. creating an order / Canceling and order
  Lambda Function
    - custom code
    - e.g. IceCreamOrderFunction
    - can be invoked for fulfilment

Demo: IceCreamBot
   Order Ice Cream - intent
     YOur name - slot
     Ice cream flavor - slot
     Ice cream size - slot
   Cancel order - intent
     OrderNo to cancel - slot


  AWS console -> Lex -> Create bot ->
    Creation Method: Traditional, Create a blank bot, bot name: IceCreamBot,
    IAM: Create a role with .. lex permissions,
    COPPA: no,
    -> Next -> 
    Language: English (US), voice interaction: Danielle 
    -> Done

    Intent: WelcomeIntent
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
    -> Build

    Test

    -> intents -> Add intent -> Add empty intent ->
    Intent Name: CreateOrderIntent -> add
    Sample Utterance: I want to order, I would  like to order, I want to buy an ice cream, 
         I would like to place an order
    Initial responce -> I can help you order ice cream
    Slots -> 
      Add Slots -> Name: name, Slot Type: Amazon.FirstName, Prompt: What is your  name -> add
      Add Slots -> Name: flavor, Slot Type: Amazon.AlphaNumeric, Prompt: What flavor -> add
         Advanced options -> Bot elicits information -> more prompt options -> 
           Slot prompts -> Add -> 
      Title: Which flavor would you like to order?
      buttons -> add button ->
      Button 1 title: Chocolate,  Value: chocolate
      Button 2 title: Strawberry, Value: strawberry
      Button 3 title: Vanilla,    Value: vanilla
      -> Update responses ->  update slots

      -> back to intent list (3) -> Add slot types -> Add blank slot type -> Name: SizeSlotType -> add
         -> SizeSlotType -> Slot type values -> 
           value: small  -> add value: 
           value: medium  -> add value: 
           value: large  -> add value: 
           -> save slot typeIt was a pleasure helping you {name}.

     -> intents -> CreateOrderIntent -> slots -> 
      Add Slots -> Name: size, Slot Type: SizeSlotType, Prompt: which size: Small/Medium/Large? -> add

     -> Confirmation -> 
        Confirmation prompt: Ready to place order for {size} {flavor} ice cream. Please confirm - Yes/No
        Decline Response: Okay, order not placed.
        Fulfillment:
          On successful fulfillment: Order successfully placed
          In case of failure: Oops! Something went wrong. Please try again later.
        Closing response:
          message: It was a pleasure helping you (name).
          -> save intent
          -> Build
          -> Test

    Intent Name: CancelOrderIntent -> add
    Sample Utterance: I want to cancel order, Cancel order, I would like to cancel order,
          Please cancel my order, Can you cancel my order,
          # additional utterance atfter "OrderNo" slot was created:
          I want to cancel order {OrderNo}, Can you cancel the order {OrderNo}


    Initial responce -> I can help you cancel your order
    Slots -> 
      Add Slots -> Name: OrderNumber, Slot Type: Amazon.AlphaNumeric, Prompt: Please provide order number -> add
     -> Confirmation -> 
        Confirmation prompt: Ready to cancel order {OrderNo}. Please confirm - Yes/No
        Decline Response: Okay, not canceling order.
        Fulfillment:
          On successful fulfillment: Your Order {OrderNo} was canceled
          In case of failure: Oops! Something went wrong. Please try again later.
          -> Advanced Options -> uncheck: Use Lambda function
          -> save intent
          -> Build
          -> Test

     -> intents -> FallbackIntent -> slots -> 
       User request acknowledgement -> 
          Mesage group -> Mesage: ...
          set values -> Next step in conversation: Intent: Intent: welcomeIntent
          -> update intent
          -> Save intent
          -> Build
          -> Test

     # create 10% discount for large chocolate
     -> intents -> CreateOrderIntent -> slots -> 
        -> Slots -> slot: size -> Advanced Options -> Slot capture: success response -> set values -> 
          Next Step in Converation -> Evaluate conditions
            -> "+ additional conditional branching" ->  change "branch1" to: "Discount_10percent"
               condition: if {flavor} = "chocolate" && {size} = "large"
               -> response -> Message Group -> Message:  A {size} {flavor} ice cream qualifies for a 10% discount!!
               -> set values -> Session Attributes: [discount] = "10"
               -> Next step in conversation: Confirm Intent
            -> default branch
               -> set values -> Session Attributes: [discount] = "0"
                 

 AWS Console -> Lambda -> Create Function -> Author from Scratch ->
   Name: iceCreamOrderFcn, Runtime: Python 3.9, Create New Role with Lambda permissions,
     Create a new role with basic Lambda permssions
     -> Create Function

Lambda function: iceCreamOrderFunction

lambda code:
   +++++++++++++++

import json

def prepareResponse(event, msgText):
    response = {
        "sessionState": {
          "dialogAction": { "type": "Close" },
        "intent": {
          "name": event['sessionState']['intent']['name'],
            "state": "Fulfilled"
           }
         },
         "messages": [ { "contentType": "PlainText", "content": msgText } ]
     }
    return response

def cancelIceCreamOrder(event):
    # Your order cancelation code here
    msgText = "Order has been canceled"
    return prepareResponse(event, msgText)

def createIceCreamOrder(event):
    firstName      = event['sessionState']['intent']['slots']['name']['value']['interpretedValue']
    iceCreamFlavor = event['sessionState']['intent']['slots']['flavor']['value']['interpretedValue']
    iceCreamSize   = event['sessionState']['intent']['slots']['size']['value']['interpretedValue']

    print(f"firstName: {firstName}, iceCreamFlavor: {iceCreamFlavor}, iceCreamSize: {iceCreamSize}")

    discount = event['sessionState']['sessionAttributes']['discount']
    print(f"discount: {discount}")

    # Your order creation code here.
    msgText = "Your Order for , " + str(iceCreamSize) + " " + str(iceCreamFlavor) + " Ice Cream has been place with Order #: 342342"

    return prepareResponse(event, msgText)


def lambda_handler(event, context):
    debug = False
    intentName = event['sessionState']['intent']['name']
    response = None
    print (f"intentName: {intentName}")
    if debug == True:
        print (f"\nevent: \n{event}\n")

    if intentName == 'CreateOrderIntent':
        response = createIceCreamOrder(event)
    elif intentName == 'CancelOrderIntent':
        response = cancelIceCreamOrder(event)
    else: 
        raise Exception('The intent : ' + intentName + ' is not supported')
    return response

   +++++++++++++++


Calling Lambda Function from Lex:
     AWS -> Lex -> IceCreamBot -> Intents ->
     -> intents -> CreateOrderIntent -> Fulfillment -> On successful fulfillment -> Advanced options ->
       select "Use a Lambda function for fulfillment" -> update options -> Save Intent

     -> intents -> CancelOrderIntent -> Fulfillment -> On successful fulfillment -> Advanced options ->
       select "Use a Lambda function for fulfillment" -> update options -> Save Intent

       -> Build

       -> Test -> settings <icon top left> -> Lambda function -> Source: iceCreamOrderFcn, version: $LATEST
           -> Save



Create a Bot Version from Draft version:
   For deploy, lambda function needs to be specified in the Alias
    
        Note: Only Draft Version exists so far
        Note: IAM role used: AWSServiceRoleForLexV2Bots_35D2ORMUDKR
     AWS -> Lex -> IceCreamBot -> Bot Versions -> Create Version 
        -> Create   # creates Version 1

     # need to create Alias to associate with Version 1

      # note: current only TestBotAlias exists, and it is only intented for testing
     AWS -> Lex -> IceCreamBot -> Deployment -> Aliases -> Create alias ->
        Name: iceCreamBotAlias, Associate with a version: Version 1 -> Create
      
     AWS -> Lex -> IceCreamBot -> Deployment -> Aliases -> iceCreamBotAlias -> Lanaguage -> English ->
        Source: iceCreamOrderFcn, version: $LATEST -> Save
       
       

