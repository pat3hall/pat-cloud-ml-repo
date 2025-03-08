#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import uuid

ENTITY_TYPE    = "dojoentity"
EVENT_TYPE     = "dojo-event" 

DETECTOR_NAME = "dojodetector"
DETECTOR_VERSION  = "1"

eventId = uuid.uuid1()

fraudDetector = boto3.client('frauddetector')

response = fraudDetector.get_event_prediction(
detectorId = DETECTOR_NAME,
eventId = str(eventId),
eventTypeName = EVENT_TYPE,
eventTimestamp = '2020-07-13T23:18:21Z',
entities = [{'entityType':ENTITY_TYPE, 'entityId':str(eventId.int)}],
eventVariables = { 'email_address' : 'johndoe@exampledomain.com', 'ip_address' : '192.10.10.24'})

print(response)


# In[ ]:




