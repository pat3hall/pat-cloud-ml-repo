import requests
import boto3
import uuid
import time
import random
import json
import sys


kinesis_stream="my-data-stream"
kinesis_arn="arn:aws:kinesis:us-east-1:012345678910:stream/my-data-stream"
region='us-east-1'

client = boto3.client('kinesis', region_name=region)
partition_key = str(uuid.uuid4())


# Added 08/2020 since randomuser.me is starting to throttle API calls
# The following code loads 500 random users into memory
number_of_results = 500
r = requests.get('https://randomuser.me/api/?exc=login&results=' + str(number_of_results))
data = r.json()["results"]

cntmsg = 0
errcnt = 0
errmax = 3
print (f'\nStarting producer put_record stream\n')
while True:
    # The following chooses a random user from the 500 random users pulled from the API in a single API call.
    random_user_index = int(random.uniform(0, (number_of_results - 1)))
    random_user = data[random_user_index]
    random_user = json.dumps(data[random_user_index])
    response = client.put_record(
        StreamName=kinesis_stream,
        StreamARN=kinesis_arn,
        Data=random_user,
        PartitionKey=partition_key)
    cntmsg += 1

    print('Message sent #' + str(cntmsg))

    # If the message was not sucssfully sent print an error message
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('\nError!\n')
        print(response)
        errcnt += 1
        if ( errcnt > errmax ):
               print (f'\nEXITING: Error Count > Error Max {errmax}\n')
               sys.exit(1)
    time.sleep(random.uniform(0, 1))
