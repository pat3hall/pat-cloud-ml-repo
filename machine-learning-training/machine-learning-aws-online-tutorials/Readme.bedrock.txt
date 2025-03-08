-------------------------------------------------------------
This file contains information on AWS Machine Learning Exam Tools

-------------------------------------------------------------
Generatative AI
  https://www.ourcrowd.com/learn/generative-ai-vs-traditional-ai
  - Generative AI is an artificial intelligence that is capable of generating outputs such as text, images, 
    and other data. 
  - This type of AI works primarily by taking large quantities of existing data, analyzing it, and, based on 
    those findings, producing new content. 
  - What this means is that generative AI relies on machine learning to recognize, predict, and create content 
    using the data sets it can access. 

Characteristics of Generative AI
  Neural network generators 
    – Generative AI utilizes neural networks like GANs and VAEs to generate relevant and original output.
  Varied applications 
    – These AI models are adaptable and versatile, producing different types of content. 
    - They have a wide range of applications across a large number of industries. 
  Creating new content using prompts 
    – Generative AI uses data and prompts to create new content instead of simply analyzing existing data. 
    - The output is unique and relevant to the prompt used to generate it. 


####################################################################

Most popular generative AI applications:
  https://www.nvidia.com/en-us/glossary/generative-ai/

  Language: 
    - Text is at the root of many generative AI models and is considered to be the most advanced domain. 
    Large Language Models (LLMs)
    - One of the most popular examples of language-based generative models are called large language models (LLMs). 
    - Large language models are being leveraged for a wide variety of tasks, including essay generation, code 
      development, translation, and even understanding genetic sequences.
  Audio: 
    - Music, audio, and speech are also emerging fields within generative AI. 
    - Examples include models being able to develop songs and snippets of audio clips with text inputs, recognize 
      objects in videos and create accompanying noises for different video footage, and even create custom music.
  Visual: 
    - One of the most popular applications of generative AI is within the realm of images. 
    - This encompasses the creation of 3D images, avatars, videos, graphs, and other illustrations. 
    - There’s flexibility in generating images with different aesthetic styles, as well as techniques for editing 
      and modifying generated visuals. 
    - Generative AI models can create graphs that show new chemical compounds and molecules that aid in drug discovery, 
      create realistic images for virtual or augmented reality, produce 3D models for video games, design logos, 
      enhance or edit existing images, and more.
  Synthetic data: 
    - Synthetic data is extremely useful to train AI models when data doesn’t exist, is restricted, or is simply 
      unable to address corner cases with the highest accuracy. 
    - The development of synthetic data through generative models is perhaps one of the most impactful solutions 
      for overcoming the data challenges of many enterprises. 
    - It spans all modalities and use cases and is possible through a process called label efficient learning. 
    - Generative AI models can reduce labeling costs by either automatically producing additional augmented 
      training data or by learning an internal representation of the data that facilitates training AI models with 
      less labeled data.

AWS Bedrock

  - Amazon Bedrock is a fully managed generative AI service 
  - offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, 
    Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon through a single API, along with a 
    broad set of capabilities you need to build generative AI applications with security, privacy, 
    and responsible AI. 
  - can easily experiment with and evaluate top FMs for your use case, privately customize them with 
    your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build 
    agents that execute tasks using your enterprise systems and data sources.

####################################################################
Integrating Generative AI Models with Amazon Bedrock
  Mike Chambers
  https://www.youtube.com/watch?v=nSQrY-uPWLY

  Bedrock User Guide
    https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html

 AWS Console -> Bedrock -> Bedrock configurations <left tab> -> Model Access
    - provides a list of available [foundation] base models with access status; 
    - you need to request access to all models or specific models before they are available

 AWS Console -> Bedrock -> Foundation Models <left tab> -> Base Models
   - provides a list of foundation models along with descriptions, etc

 AWS Console -> Bedrock -> Getting Started <left tab> -> providers
   - provides a list of foundation model provides along with descriptions of the foundational models

 AWS Console -> Bedrock -> Playgrounds <left tab> -> [chat Text Image]
    -> provides models by the type of use case [playground]
       - foundation models can be used to:
          - generate text
            - generate text from text
          - generate images
            - generate text from images
          - embeddings modesl

  Programming language support
    - includes Javascript, Python, any languauge that has a supporting SDK or directly with the API

  Foundation Model API
    - passed to python models as kwargs with 'body' passed as a json string (e.g. json.dump()
    - Example model API is found at: 
          AWS Console -> Bedrock -> Getting Started <left tab> -> providers -> <provider> -> <model> -> API request
       

Code:  Example Python code to access Bedrock Antropic Claude-3-haiku mode
++++++++++++++++++++++++++
import boto3
import json
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

prompt = "What is the capitcal city of Australia"
# note: using json.dumps() for body section since it must be sent to Auntropic claude 3 model as a json string
kwargs = {
  "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
  "contentType": "application/json",
  "accept": "application/json",
  "body": json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": prompt
          }
        ]
      }
    ]
  })
}

# invode_model is the single API to invoke any bedrock fundational models - kwwargs specify the specific model and it specific inputs
#   kwags 'modelId' specifies the specific bedrock model to use
response = bedrock_runtime.invoke_model(**kwargs)

body = json.loads(response['body'].read())

print(body)
++++++++++++++++++++++++++


  https://community.aws/generative-ai
    - to find Generative Articles/blogs
  

####################################################################

Build an AWS Solutions Architect Agent with Amazon Bedrock
  https://www.youtube.com/watch?v=BuDkY-P5JZA

  Github:
    https://github.com/build-on-aws/amazon-bedrock-agents-quickstart

  overview:
    - shows you how to create an AWS Solutions Architect Agent tailored to your needs, using Amazon Bedrock. 
    - Combining large language model (LLM) capabilities with customized tools, Amazon Bedrock empowers your AI 
      agent to create and deploy well-architected solutions on AWS.

   Agent
     - application powered by a Large Language Model (LLM) with predefined instructions on how to act
     - agent has a set of tools which are self contained designed to perform a specific task
     - demo agent can help customers build solutions on a AWS
       - well teach agent to learn AWS best practices and sight them



####################################################################
Install Amazon CodeWhisperer and Build Applications Faster Today (How To)
  Note: CodeWhisperer has been merged with Amazon Q
  https://www.youtube.com/watch?v=sFh3_cMUrMk


  Visual Studio Code ->
      Marketplace <left pane icon>  -> search "AWS Toolkit" -> install 
        -> Once installed, AWS logo will be on left pane
        -> Open Amazon Q (bottom of VS window)
           -> verify profile (aws config)
           -> log in the AWS Community / Builder ID
        
        -> Amazon Q icon will now be in left pane

        -> if Amazon Q is not giving you suggestions, then uses 

####################################################################
Installing Amazon Q in JupyterLab
  https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/jupyterlab-setup.html

    # find version
    pip show jupyterlab

    #Installation using pip for Jupyter Lab version >= 4.0
    pip install amazon-q-developer-jupyterlab-ext

    ERROR: Could not build wheels for amazon-q-developer-jupyterlab-ext, which is required to install pyproject.toml-based projects


LangChain
  https://aws.amazon.com/what-is/langchain/
  - open source framework for building applications based on large language models (LLMs). 
  - LLMs are large deep-learning models pre-trained on large amounts of data that can generate responses to user queries—for 
    example, answering questions or creating images from text-based prompts. 
  - LangChain provides tools and abstractions to improve the customization, accuracy, and relevancy of the information the models 
    generate. For example, developers can use LangChain components to build new prompt chains or customize existing templates. 
  - LangChain also includes components that allow LLMs to access new data sets without retraining.
  
Why is LangChain important?
  - LLMs excel at responding to prompts in a general context, but struggle in a specific domain they were never trained on. 
  - Prompts are queries people use to seek responses from an LLM. 
  - For example, an LLM can provide an answer to how much a computer costs by providing an estimate. 
  - However, it can't list the price of a specific computer model that your company sells. 

  - To do that, machine learning engineers must integrate the LLM with the organization’s internal data sources and apply 
    prompt engineering—a practice where a data scientist refines inputs to a generative model with a specific structure and context. 


####################################################################
Agents Tools & Function Calling with Amazon Bedrock (How-to)
  Mike Chambers
  https://www.youtube.com/watch?v=2L_XE6g3atI

  AWs Console -> Bedrock -> Playground <left tab> -> Text -> Select Model -> Anthropic -> Claude 2.1 

  prompt: What is the time?
    Sorry, I don't have access to the current time. ...
  # does not know since it is not trained on live data
  # to actually handle time, it needs access to a clock

  prompt: What is 10 plus 20?
  10 plus 20 equals 30
  # has a rudimentry sense of mathematics
  # to actually handle mathematics, it needs access to a calculator

  Agents
    - give LLMs access to tools they can use
    Action Groups
      - contain the tools that we want to give to the agent

  AWs Console -> Bedrock -> Orchestration <left tab> -> Agent -> Create Agent
    Name: demo-agent -> Create 

    -> demo-agent -> Edit in Agent Builder
       Create an IAM role
       Select Model: Anthropic Claude V2.1   # recommends also trying Claude 3.1
       Instructions for the Agent (required): You are a helpful agent who can perform basic math and tell the time.
       -> Save and Exit <top>
    -> demo-agent -> Edit in Agent Builder
       Action Group -> Add
         Action Group Name: demo-action-group
            Action group type: Define with function details, Action Group Invocation: Quick create a new Lambda function
              # creates a stub lambda function for the logic to be run (e.g. add numbers, telling time, etc.)
              Action group function 1: Name: get_time, description: gets the current time in UTC, 
              # parameters can be sent to the funcion, if any parameters specified, LLM will look for them in the conversation
              paramters: <skip -> no parameters needed in this case>
              -> click on "add another funciotn
              Action group function 2: Name: add_two_numbers, description: add two numbers and return the result,
              # Note: parameter description will be used as part of the prompt
              paramters: 
                 -> add parameter -> Name: number_1, description: First number to be added., type: Number, Required: True 
                 -> add parameter -> Name: number_2, description: Second number to be added., type: Number, Required: True 
                 -> add # takes a minute or so because it is creating a Lambda function with stub code plus Lambda fcn permissions

                -> scroll down to the lamba function name and click view 
                # in Lambda function

Code: lambda function generated stub code:
  #+++++++++++++++++++++++
import json
def lambda_handler(event, context):
    agent = event['agent']
    actionGroup - event[actionGroup']
    function = event['function']
    parameters = event.get('parameters',[])

    # Execute your business loge here. For more information, refere to: http:
    responseBody = {
        "TEXT": {
            "body": "The function {} was called successfully!".format(function)
        }
    }

    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': 
            'responseBody': responseBody
         }

  #+++++++++++++++++++++++

Code: lambda function implemented code:
  #+++++++++++++++++++++++
import json
def lambda_handler(event, context):
    agent = event['agent']
    actionGroup - event[actionGroup']
    function = event['function']
    parameters = event.get('parameters',[])

    # Function to get the current time
    def get_time():
        return datetime.datetime.utcnow().strftime('%H:%M:%S')

    # funcion to add two numbers
    def add_two_numbers(return number_1, number_2):
        return number_1 + number_2

    # Extracting values from parameters
    parm_dict = {param['name'].lower(): int(param['value']) for param in parameters if param['type'] == 'number'}

    # check the function name and execute the corresponding action
    if function == "add_two_numbers":
        # safe extraction of the number_1 and number_2 from parameters
        number_1 = parameter_dict.get('number_1')
        number_2 = parameter_dict.get('number_2')

        # Ensure both numbers are provided are of the correct type
        if number_1 is not Nome and number_2 is not None:
            try:
                number_1  = int(number_1)
                number_2  = int(number_2)
                result = add_two_numbers(number_1, number_2)
                result_text = "The result of adding {} and {} is {}".format(number_1, number_2, result)
            expect ValueError:
                result_text = "Error: Non-integer parameters."

        else:
            result_text = "Error: Missing one or more required parameters."

        responseBody = {
            "TEXT": {
                "body": result_text
            }
        }

    elif function == "get_time":
        result = get_time()
        result_text = "The time is {}".format(result)

        responseBody = {
            "TEXT": {
                "body": result_text
            }
        }


    else:
        responseBody = {
            "TEXT": {
                "body": "The function {} was called successfully!".format(function)
            }
        }

    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': 
            'responseBody': responseBody
         }

   #+++++++++++++++++++++++

   -> AWS -> Lambda -> demo-action-group-* -> insert updated code from above -> deploy
   
   # back to bedrock

   -> AWS -> Agents -> demo-agent -> save and exit <top center>
      # prepare agent for testing
      -> Prepare <top center>
      # test agent <left side pane>
         Enter your message here: What is the time?
         # returns:
         The time is <search_result> The time is ....</search_result>
      
         Enter your message here: What is 10 plus 123?
         # returns:
         10 plus 123 is <search_result> The result of adding 10 and 123 is 133</search_result>
         -> show trace -> Preprocessing trace <top left>
            -> near bottom of trace it shows it used calling agent using hte "demo-action-group::add_two_numbers function ...
         -> show trace -> Orchestration and Knowledge base <top 2nd from left>
            -> near bottom, it show the invocationInput for the "demo-action-group" including parameter and the output
      

  Provided links:
    Automate tasks in your application using [bedrock] conversational agents
    https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html

    Youtube #amazonBedrock
    https://www.youtube.com/hashtag/amazonbedrock

####################################################################

  Bedrock Agents
    Automate tasks in your application using conversational agents
    https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html

    Agents perform the following tasks:
      - Extend foundation models to understand user requests and break down the tasks that the agent must perform 
        into smaller steps.
      - Collect additional information from a user through natural conversation.
      - Take actions to fulfill a customer's request by making API calls to your company systems.
      - Augment performance and accuracy by querying data sources.
    

To use an agent, you perform the following steps:
    1. (Optional) Create a knowledge base to store your private data in that database. 

    2. Configure an agent for your use case and add at least one of the following components:
      - At least one action group that the agent can perform. 
      - Associate a knowledge base with the agent to augment the agent's performance. 

    3. (Optional) To customize the agent's behavior to your specific use-case, modify prompt templates for the 
       pre-processing, orchestration, knowledge base response generation, and post-processing steps that the agent performs. 

    4. Test your agent in the Amazon Bedrock console or through API calls to the TSTALIASID. 

    5. When you have sufficiently modified your agent and it's ready to be deployed to your application, create 
       an alias to point to a version of your agent. 

    6. Set up your application to make API calls to your agent alias.

    7. Iterate on your agent and create more versions and aliases as necessary.

    
####################################################################

Amazon Bedrock: Your Top Questions Answered
  Mike Chambers
  https://www.youtube.com/watch?v=PdbvTyRTlVY

  Bedrock
    - a collection of foundation models (FM) that you can combine through a single API
    - includes text generatation and image generation FM from a number of providers


 AWS Console -> Bedrock -> Foundation Models <left tab> -> Base Models
   - provides a list of foundation models along with descriptions, etc
     - model providers include AI21 Labs, Amazon, Antrophic, Cohere, Meta, Mistral, & Stability AI


 AWS Console -> Bedrock -> Playground <left tab> -> Chat -> Anthropic -> Claude V2
       context size = up to 100K # very large context, so it can take in a large amount of information
     Inference Configuration (model Parameters): 
        temperature: 
          - randomness of the model response; the higher the temperature, the more creative the model output
        top P & top K: 
          - also influence the randomness and diversity of the model output
        maximum length:  
          - maximum number of tokens that will be generated for output
          - useful from a cost-optimization
        stop sequences:  
          - if model produces this sequence, we want it to stop; start prefilled
      Write something prompt:
         -> can enter question/information to model

        View API request:
           - provides a code snippet that can be uses when working with the API directly

####################################################################

Amazon Bedrock: What Are Tokens?
  Mike Chambers
  https://www.youtube.com/watch?v=y6IRvdxd6KA


  Tokens
    - basis for how on-demand pricing works
    - charge for the tokens going into the model and tokens going out of the model

  Tokenizer
    - takes the input text and breaks it up into tokens and applies a number to each token
    - words and punctuation could be broken up into tokens
    - words could be broken into multiple tokens (e.g. bedrock -> tokens: bed, rock 
    - tokenization is model specific
      

  Finding number of tokens:
    Cloudwatch -> Metric -> Bedrock -> By ModelID -> <model> -> inputTokenCount -> Enable <select> -> examine graph

  Log Bedrock model invocations
    Cloudwatch -> Logs -> Log Groups -> Create Log Group -> 
       Log Group name; bedrock-log-group -> Create

    Bedrock -> Bedrock Configurations <left tab near bottom> -> Settings -> Model Invocation logging ->
      enable "Model invocation logging"
         # Note: can log to "S3 only", "Cloudwatch logs only", or "Both S3 and CloudWatch logs"
         select "CloudWatch Logs Only", log group name: bedrock-log-group, create and use a new role, 
           Service role name: bedrock-logging-role 
           -> Save settings


####################################################################

Which fundation models to choose in AWS bedrock
    https://blog.economize.cloud/aws-bedrock-foundation-models-list/


  How do Foundation Models work in AWS Bedrock?
    - serverless
    - Single API for access to all models
    - At the heart of Amazon Bedrock is its ability to streamline the experimentation and evaluation of leading FMs, 
      tailored for specific use cases. 
    - Users can fine-tune these models with their data, employing techniques like fine-tuning and Retrieval Augmented 
      Generation (RAG) with models like Amazon Q to create customized solutions that resonate with their unique business 
      requirements. 

  Retrieval Augmented Generation (AUG) in Bedrock
    - RAG is a powerful technique used to enrich FM responses with contextual and relevant company data


  Model Customization
   - Customization is a cornerstone of Amazon Bedrock
   - Techniques like fine-tuning and continued pre-training enable users to refine models using labeled or 
     unlabeled data from Amazon S3, creating private, tailored versions of the base models.
   Fine Tuning
     - Supported models like Cohere Command, Meta Llama 2, and various Amazon Titan models can be fine-tuned using labeled datasets.
   Continued Pre-training 
     – Continued pre-training allows for domain-specific adaptations of Amazon Titan Text models. 


Bedrock Foundation Model
format:

    FM Provider
       Model Version
       Max Capacity 
       Distinct Features & Languages 
       Supported Use Cases & Applications 
       Pricing


     AI21 Labs Jurassic 
       Jurassic-2 Ultra 8,192 tokens
         Advanced text generation in English, Spanish, French, German, Portuguese, Italian, Dutch 
         Intricate QA, summarization, draft generation for finance, legal, and research sectors 
         $0.0188 per 1,000 tokens
     
       Jurassic-2 Mid 8,192 tokens
         Text generation in multiple languages for broad applications 
         Ideal for QA, content creation, info extraction across various industries 
         $0.0125 per 1,000 tokens
     
     Anthropic Claude 
        Claude 2.1 
          200K tokens 
          High-capacity text generation, multiple languages 
          Comprehensive analysis, trend forecasting, document comparison 
          $0.00800 input, $0.02400 output per 1,000 tokens
        Claude 2.0 
          100K tokens 
          Creative content generation, coding support, multiple languages 
          Versatile for creative dialogue, tech development, and educational content 
          $0.00800 input, $0.02400 output per 1,000 tokens
        Claude 1.3 
          100K tokens 
          Writing assistance, advisory capabilities, multiple languages 
          Effective for document editing, coding, general advisory in diverse sectors 
          $0.00800 input, $0.02400 output per 1,000 tokens
        Claude Instant 
          100K tokens 
          Rapid response generation, multiple languages 
          Fast dialogue, summary, and text analysis, ideal for customer support and quick content creation
          $0.00163 input, $0.00551 output per 1,000 tokens
     
     Cohere Command & Embed 
       Command 
         4K tokens 
         Advanced chat and text generation, English
         Dynamic user experiences in customer support, content creation for marketing and media
         $0.0015 input, $0.0020 output per 1,000 tokens
       Command Light 
         4K tokens   
         Efficient text generation, English
         Cost-effective for smaller-scale chat and content tasks, adaptable for business communications
         $0.0003 input, $0.0006 output per 1,000 tokens
       Embed – English 
         1024 dimensions 
         Semantic search, English
         Ideal for precise text retrieval, classification in knowledge management and information systems
         $0.0001 per 1,000 tokens
       Embed – Multilingual 
         1024 dimensions 
         Global reach with support for 100+ languages
         Multilingual applications in semantic search and data clustering for international business and research 
         $0.0001 per 1,000 tokens
     
     Meta Llama 2 
       Llama-2-13b-chat 
         4K tokens
         Optimized for dialogue, English
         Small-scale tasks like language translation, text classification, ideal for multilingual communication platforms 
         $0.00075 input, $0.00100 output per 1,000 tokens
       Llama-2-70b-chat 
         4K tokens
         Enhanced for large-scale language modeling, English
         Suitable for detailed text generation, dialogue systems in customer service, and creative industries
         $0.00195 input, $0.00256 output per 1,000 tokens
     
     Stable Diffusion 
       SDXL 1.0 
         77-token limit for prompts
         Native 1024×1024 image generation, English
         High-quality image creation for advertising, gaming, and media, excels in photorealism 
         Standard: $0.04, Premium: $0.08 per image (1024×1024)
       SDXL 0.8 
         77-token limit for prompts 
         Text-to-image model, English
         Creative asset development in marketing, media, suitable for diverse artistic styles
         Standard: $0.018, Premium: $0.036 per image (512×512)
     
     Amazon Titan 
       Titan Text Express 
         8K tokens
         High-performance text model, 100+ languages
         Diverse text-related tasks in content creation, classification, and open-ended Q&A, applicable in education and content marketing
         $0.0008 input, $0.0016 output per 1,000 tokens
       Titan Text Lite 
         4K tokens
         Cost-effective text generation, English
         Efficient for summarization, copywriting in marketing and corporate communications
         $0.0003 input, $0.0004 output per 1,000 tokens
       Titan Text Embeddings
         8K tokens
         Text translation to numerical representations, 25+ languages
         Semantic similarity, clustering for data analysis, knowledge management
         $0.0001 per 1,000 tokens
       Titan Multimodal Embeddings 
         128 tokens, 25 MB images
         Multimodal (text and image) search, English
         Accurate and contextually relevant search, recommendation experiences in e-commerce, and digital media
         $0.0008 per 1,000 tokens; $0.00006 per image
       Titan Image Generator 
         77 tokens, 25 MB images 
         High-quality image generation using text prompts, English
         Image creation and editing for advertising, e-commerce, and entertainment with natural language prompts
         512×512: $0.008, 1024×1024: $0.01 per image

####################################################################

Choose a Foundation Model For Your Use Case Pt1 - AWS ML Heroes in 15 
  Tomaasz Dudek
  https://www.youtube.com/watch?v=_4HFgLTkX2Y

  Bedrock Overview
    - Easily build with Foundation Models (FM) (FMaaS)
    - Securely build generative IA apps with your data
    - Deliver customized experiences using your data
    - enable generative AI apps to complete task with agents

  Bedrock core API: InvokeModel
    - use a single API to securely access models provide AWS and other AI companies or your own ones
    - handles text-to-text, text-to-image, and image-to-image, and more
    - fully managed experience

  Agents for Bedrock
    - write down all the external APIs that could be useful while solving the task
    - build generative AI-powered apps that complete tasks with agents

  Security
    - data remains in the region where the API call is processed
    - data is encryptyed in transit and at rest
    - no data is used to retrain the original base FMs
    - integrated with CloudTrail and PrivateLink

  Customized experiences using your data
    - fine-tune FM for a particular task without having large volumes of data or highly-specialized team
    - when you fine-tune a FM, it creates a separate, private copy of the base FM, accessible only to you
    - Supplement organization-specific information to the FM
    - deliver customized search capabilities for your organization

  Foundation Model Providers:
    Amazon Titan
      Titan
    AI21labs 
      Jurasic-2
    Anthropic
      Claude 2
    cohere
      command and Embed
    stability.ai
      Stable Diffusion

  Stability.ai
    - leading open-source multimodel generative AI company
    - builds Foundation Models across multimodalities including:
       - image, language, audio, video, 3D, biology
    Foundation Models
      SD XL 1.0
      - most recent FM is Stable Diffusion XL 1.0 (SD XL 1.0)
      - poser text-to-image and image-to-image diffusion model
      Use Case
        Advertising & Marketing
          - iterate on images and design using text prompts
          - create infinite version of marketing material
            - use image model to create customized images for marketing and advertising campaigns
              or personalized ads
          - localize marketing campaigns
        Media & Entertainment
          - augment create teams
          - create, reimagine, and ideate
          - fine-tune models on individual, scenes, and objects to create infinite creative assets
          - Create new characters, new scenes, and storyboard ideas
        Gaming & Metaverse
          - Create new characters, scenes, and whole new worlds
          - localize themes and assets
          - add words and text into your image outputs
          
      Stable Beluga 1 and Stable Beluga 2
        - latest language model (not included in Bedrock)
        - open source language model

    Getting started with Stability.ai
      Stability Platform API
        - managed serverless 
      Bedrock
        - managed serverless 
      Amazon SageMaker JumpStart
        - provides more control over their model
      Stability.ai FMs Available on EC2 Inf2 Instances
        AWS Inferentia2
          - high performance at the lowest cost per inference for LLMs and diffusion models
        


####################################################################

Choose a Foundation Model For Your Use Case (part 2) - AWS ML Heroes in 15 
  Tomaasz Dudek
  https://www.youtube.com/watch?v=RHO9AnspStg


  Bedrock
    - foundation model as a service
    - easily build with FMs (FMaaS)
    - securely build generative AI apps with your data
    - deliver customized experiences using your data
      - fine-tune the existing models using your datasets

  Bedrock core API: InvokeModel
    - use a single API to securely access models provide AWS and other AI companies or your own ones
    - handles text-to-text, text-to-image, and image-to-image, and more
    - fully managed experience

  Agents for Bedrock
    - allows you to define even a complex task and define a set of APIs that can be used to solve that task
    - write down all the external APIs that could be useful while solving the task
    - build generative AI-powered apps that complete tasks with agents

  Security
    - data remains in the region where the API call is processed
    - data is encryptyed in transit and at rest
    - no data is used to retrain the original base FMs
    - integrated with CloudTrail and PrivateLink

  Customized experiences using your data
    - fine-tune FM for a particular task without having large volumes of data or highly-specialized team
    - when you fine-tune a FM, it creates a separate, private copy of the base FM, accessible only to you
    - Supplement organization-specific information to the FM
    - deliver customized search capabilities for your organization

  Foundation Model Providers:
    Amazon Titan
      Titan
    AI21labs 
      Jurasic-2
    Anthropic
      Claude 2
    cohere
      command and Embed
    stability.ai
      Stable Diffusion

   AI21Labs
      Large Language models (LLMs) products
        Jurasic Series
          - instruction following meaning that you can give them instructions in natural language 
            and they will follow
          - multilingual
          Jurasic-2 Light
          Jurasic-2 Mid
            - tradeoff between quality and cost/latency (speed)
          Jurasic-2 Ultra
            - most capable and powerful model
            - best output quality 
        Access
          - via AI21Studio developer platform
          - AWS access through Bedrock (Jurasic-2 Mid & Ultra) and Sagemaker
        Python access [Bedrock] example summarizing description
          import ai21
          brand = "ASUS"
          name = "ASUS 16\" Vivobook 16k Labtop (Black)"
          features = "...."
          prompt = f""""Sile Brand Guidelines ...."""
          ....
          response = ai21.Completion.execute(destination=1i21.bedrockDestination(model_id=ai21.BedrockModelID.32_ULTRA),
                                            prompt=prompt, maxTokens=200, temperature=1,topP=0.9,stopSequences=['##'])
          print(response['completions][0]['data']['text'])

        Jurasic Use Cases include:
          - generating product descriptions
          - summarizing financial documents
          - generating outlines and full posts
          - extracting entities and insights from long reports
          - analyze reviews
          - automate customer support process
        


####################################################################

Improve your Generative AI Application with RAG
  Mike Chambers & Tiffany ???
  https://www.youtube.com/watch?v=FQhksZ87Ncg
  Summary:
    In this video, Mike and Tiffany explain the fundamental principles of RAG to avoid hallucinations from your LLM applications.


   Bedrock Console implementation for AWS Service Training recommendations apps based on your interests, etc.
      - add HTML files on AWS services to S3 bucket and provided S3 bucket to Bedrock (???)

   Retrival Augmented Generation (RAG)
     - a way to custom your application by having adding knowledge base via RAG
     - a way to give your application access to your specific information or current information
       without fine-tuning or retraining the model
     - creates a vector database in the background which contains your documents, and then your app uses
       ability to search this vector database [e.g. retrival] to augment the AI generation
       
     Bedrock models
       large Language model
       embedding models
         - take text and convert them into  embeddings that you can for things like a [RAG] search
         Titan embed model
           - Amazon's own embedding model
     
Code: Jupyter Notebook code to implement RAG (should not ever have to code long hand like this with Bedrock!!!):
   #++++++++++++++++++++++++++
import boto3
import json
import faiss
import numpy  as np
from jinja2 import Template

# create bedrock runtime client
bedrock_runtime_client = boto3.client("bedrock-runtime", region_name="us-west-2")

# facts: These facts represent a respository of information. In the real systems, these would represent
#        a large collection of documents

facts = [
     "The Sun accounts ..."
     "Mercury is the smallest planet ..."
     ....
     "Black holes ..."
     ]

def embed(text):
    """
    Embeds a fact uinsg the Amazon titan-embed-text-v1 model.

    Args:
        fact (str): The fact to embed.

    Returns:
        list: The embedding of the fact.
    """
    kwargs = {
        "modelId": "amazon.titan-embed-text-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": json.dumps({
            "inputTest": fact
         })
    }

    resp = bedrock_runtime_client.invoke_model(**kwargs)

    resp_body = json.load(resp.get('body').read())
    return resp_body.get('embedding')

# Store part RAG:
# create an array to store the embeddings  [in a vector database]
embeddings_array = np.array([]).reshape(0, 1536)

# Embed each fact:
for fact in facts:
    embeddings = embed(fact)
    embeddings_array = np.append(embeddings_array, np.array(embeddings).reshape(1,-1),???)

# Create a vector store (FAISS is just an in memory vector store for this sample)
index = faiss.IndexFlatL2(1536)

# add out vectors to the vector store. And print the number of vectors in the vector store
index.add(embeddings_array)
print(index.ntotal)


# Retrival Part of RAG:
# QUERY vector database
# set a question to ask. We also need to get an embedded version of the question
query = "What is the Milky Way?"
embedded_query = embed(query)

# using the vector store, find the facts that are the most similar to our question
k = 4 # return the 4 most relevant facts from the database
D, I = index.search(np.array(np.array([embedded_query]),k)
print(I) # <- The indexes of the relevant facts
print(D) # <- The distances of the relevant facts (or how similar they are)

[[14 13 0 11]]
[[0.  381.79772   420.1158  423.44202]]

# Augment Part of RAG
# prompt will be provided to Antrophic Claude v2
prompt_template = """

Human: Given the facts provided in the following list of facts between <facts> tags 
find the answer to the question written between the <question> tags.

<facts>
{%- for fact in facts %}
 - `{{fact}}`{% endfor %}
</facts>

<question{{question}}</question>

Provide an answer in full including parts of the question, so the answer can be understood ...
....
Just provide the answer, nothing else.

Assistant:"""


# Use Jinja to fill out our prompt template, adding in all the 

data = {
    'facts': [facts[index] for index in I[0],
    'question': query
}

template = Template(prompt_template)
prompt = template.render(data)

# preview the prompt
print (prompt)

# call Claude V2 LLM

kwargs = {
    "modelId": "anthropic.claude-v2",
    "contentType": "applicaition/json",
    "accept": "*/*",
    "body": json.dumps(
        {
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0,
        "top_k": 250,
        "top_p": 0.999,
        "stop_sequences": [
            "\n\nHumans:"
        ],
        anthropic_version": "bedrock-2023-05-31"
        }
    )
}

response = bedrock_runtime_client.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['completion']
print(generation)
   #++++++++++++++++++++++++++

####################################################################

Fine tuning Vs Pre-training
https://medium.com/@eordaxd/fine-tuning-vs-pre-training-651d05186faf

  Fine-tuning:
    - Fine-tuning employs labeled data to fine-tune the model’s parameters, tailoring it to the specific 
      nuances of a task. 
    - This specialization significantly enhances the model’s effectiveness in that particular task compared 
      to a general-purpose pre-trained model.
    - uses Labeled data (Supervised)

  Pre-training
    - Pre-training usually would mean take the original model, initialise the weights randomly, and 
      train the model from absolute scratch on some large corpora.
   Further/Continuos Pre-training
    - Further/Continuous pre-training means take some already pre-trained model, and basically apply 
      transfer learning — use the already saved weights from the trained model (checkpoint) and train 
      it on some new domain (i.e financial data).
    - This approach utilizes unlabelled data from a particular domain, enabling the Language Model (LLM) 
      to enhance its comprehension and performance in specific knowledge domains, such as finance, law, 
      or healthcare
    - uses UnLabeled data (UnSupervised)

####################################################################
Amazon Bedrock FAQs
  https://aws.amazon.com/bedrock/faqs/


  Retrieval augmented generation (RAG)
    What types of data formats are accepted by Amazon Bedrock Knowledge Bases?
      - Supported data formats include .pdf, .txt, .md, .html, .doc and .docx, .csv, .xls, and .xlsx files. 
      - Files must be uploaded to Amazon S3. 
      - Point to the location of your data in Amazon S3, and Amazon Bedrock Knowledge Bases takes care of 
        the entire ingestion workflow into your vector database.

    Which embeddings model is used to convert documents into embeddings (vectors)?
      - At present, Amazon Bedrock Knowledge Bases uses the latest version of the Amazon Titan Text Embeddings 
        model available in Amazon Bedrock. 
      - Titan Text Embeddings V2 model supports 8K tokens and 100+ languages and creates an embeddings 
        of flexible 256, 512, and 1,024 dimension size. 
   
    Model evaluation
      What is Model Evaluation on Amazon Bedrock?
        - Model Evaluation on Amazon Bedrock allows you to evaluate, compare, and select the best FM for 
          your use case in just a few short steps. 
        - Amazon Bedrock offers a choice of automatic evaluation and human evaluation. 
        - You can use automatic evaluation with predefined metrics such as accuracy, robustness, and toxicity.

    Bedrock Guardrails
      What are the safeguards available in Amazon Bedrock Guardrails?
        Guardrails allows you to define a set of policies to help safeguard your generative AI applications. 
        You can configure the following policies in a guardrail.
          Denied topics: 
            - define a set of topics that are undesirable in the context of your application. 
            - For example, an online banking assistant can be designed to refrain from providing investment advice.
          Content filters: 
            - configure thresholds to filter harmful content across hate, insults, sexual, and violence categories.
          Word filters (coming soon): 
            - define a set of words to block in user inputs and FM–generated responses.
          PII redaction (coming soon): 
            - select a set of PII that can be redacted in FM–generated responses. 
            - Based on the use case, you can also block a user input if it contains PII.

      


####################################################################
