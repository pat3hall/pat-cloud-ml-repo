#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import json
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')


# In[ ]:


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

