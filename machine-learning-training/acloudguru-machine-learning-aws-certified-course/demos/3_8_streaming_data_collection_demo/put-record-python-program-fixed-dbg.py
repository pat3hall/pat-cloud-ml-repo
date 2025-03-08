import requests
import boto3
import uuid
import time
import random
import json


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

print (f'\nStarting producer put_record stream\n')
x = range(10)
for n in x:
        # The following chooses a random user from the 500 random users pulled from the API in a single API call.
        random_user_index = int(random.uniform(0, (number_of_results - 1)))
        random_user = data[random_user_index]
        random_user = json.dumps(data[random_user_index])
        client.put_record(
                StreamName=kinesis_stream,
                Data=random_user,
                PartitionKey=partition_key)
        time.sleep(random.uniform(0, 1))
