
import random
import json
import os

# count is the number of receipt files to generate
count = int(os.getenv("FILE_COUNT") or 100)

# create a 'words' list with contents of 'words' file
#words = [word.strip() for word in open('../words').readlines()]
words = [word.strip() for word in open('/usr/share/dict/words').readlines()]

for identifier in range(count):
    # generate random float values between $1 and $1000
    amount = random.uniform(1.0, 1000)
    content = {
            'topic' : random.choice(words),
            'value' : "%.2f" % amount
    }
    with open(f"./new/receipt-{identifier}.json",'w') as f:
        json.dump(content,f)
