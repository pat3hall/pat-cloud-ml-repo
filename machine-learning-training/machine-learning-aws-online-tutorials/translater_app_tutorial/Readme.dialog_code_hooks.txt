This REadme is for:

Enabling dialog code hooks in Amazon Lex v2 | Amazon Web Services
https://www.youtube.com/watch?v=xXwBu549G18


Dialog Code Hooks
  - way of putting in a connection Lambda function within your Lex Bot
  - there can be multiple of theses with each intent and they are called during the
    back and forth of a converation within and intent
  - unlike the fulfillment code hook is only called at the end of an intent which 
    there is only one per intent

Converation steps ??? [within an intent]
  Start
  code hook
  get slot value <slotName>
  code hook
  confirmation
  code hook
  fulfillment
  end conversation

  -> select "Visual builder" (at bottom of intent page) to show steps
  


DiaglogCodeHook Types
  - each intent is made up of a series of conversation steps
  Standard
    - called between conversation steps to do things like validation
    - it runs as an independent step of the conversation flow
    - can be put between any 2 steps in the conversation up to fulfillment
    Common use cases:
      - validaing slot input
      - checking confidence score for intents, transscriptions or sentiment
      - adding additional information into the sessioin
      - changing flow of the conversation
  Elicitation
    - called within a conversation step
    - lets us handle things when we are not understanding what the user said in a
      more specificated way than just saying what?
    - it can only run in  the 'get slot value' or 'confirmation' steps
    - captures each thing the user says, not just the final value
    Common use-cases
      - adjusting responses based on the number of retries
      - checkiong transcription options [verify it was heard correctly]
      - handling special cases for slot capture

To use DiaglogCodeHook:
  - intent level dialogue code hook option must be enabled
     - with API, CreateIntent
           { "description": "string",
              "diaglogCodeHook": { "enabled": boolean}
              }, ...
     - in Lex console:
        -> at bottom of intent, with Editor:
          Code Hooks: "Use a Lambda funtion for initialiation and validation"
        -> at bottom of intent, with Visual builder:
           in the "Start" box -> click upper right corner for "start" form, 
              select "enable dialog code hook invocation"
        -> in intent, at bottom of each "prompt for slot: <slotName>" click on "Advanced options",
           then near bottom of advanced options, in "Dialog code hook", select "invoke Lambda function"
              - this will invoke the lambda function after process slot value
              - also need to set "Next step in conversation" under "Closing Response" -> "set values"
           

     |-----------|-------------------------|-------------------------------------------
     | Intent    |    Dialog code hook     |
     | Code Hook | Active  | Invoke Lambda |
     |-----------|---------|---------------|-------------------------------------------
     |   on      |    on   |      on       |    Lambda invoked
     |-----------|---------|---------------|-------------------------------------------
     |   on      |    on   |      off      |    Successful lambda invokation simulated
     |-----------|---------|---------------|-------------------------------------------
     |   on      |    off  |      on       |    NO Lambda invoked
     |-----------|---------|---------------|-------------------------------------------
     |   on      |    off  |      off      |    NO Lambda invoked
     |-----------|---------|---------------|-------------------------------------------
     |   off     |    on   |      on       |    Successful lambda invokation simulated
     |-----------|---------|---------------|-------------------------------------------
     |   off     |    on   |      off      |    Successful lambda invokation simulated
     |-----------|---------|---------------|-------------------------------------------
     |   off     |    off  |      off      |    NO Lambda invoked
     |-----------|---------|---------------|-------------------------------------------
     |   off     |    off  |      off      |    NO Lambda invoked
     |-----------|---------|---------------|-------------------------------------------


     Slot eliciation prompt
       - call lambda function inside "get slot value" step
       - captures all the back and forth on getting slot value
       - in "visual builder", go to "slot prompt" and click on settings icon to see 
         "slot elicitation prompt" options select "invoke Lambda hook code after each elicitation"
          and possibly "User can interrupt the prompt wne it is being read"
       - in "editor"  mode, go to "slot prompt", expand "Bot elicits information", click on 
          "More prompt otpions",  under prompt settings select "invoke Lambda hook code after each elicitation",
          and possibly "User can interrupt the prompt wne it is being read"


     code hook invokation rules
       - cannot have more than one code hook block off a conversation
       - cannot have back to back code hook blocks
       - cannot have a code hook after fulfillment
       - all code hooks will call the same lambda function
       - cook hooks are path, but not a step 
