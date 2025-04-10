
import os
import glob
import json
import shutil

try:
    os.mkdir('./processed')
# if dir exists, it raises 'FileExistsError' exception which is in a subclass OSError exception
# The parent expection, OSError, will work with pre-3.3 version, but th FileExistError may not
except OSError:
    print("'./processed' directory already exists")


# get list of receipt files
receipts = glob.glob('./new/receipt-[0-9]*.json')
subtotal = 0.0

for path in receipts:
    with open(path) as f:
        # load receipt JSON contents to 'content' dict
        content = json.load(f)
        subtotal += float(content['value'])
        # split receipt file 'path' to list and get last item [-1]
        # Note: ./new/receipt-1.json".split('/')[-1]  => [".", 'new', 'receipt-1.json'][-1]
        #    assigns 'receipt-1.json to 'name'
        name = path.split('/')[-1]
        # move processed receipt files from './new' directory to './processed' dir
        destination = f"./processed/{name}"
        shutil.move(path,destination)
        print(f"moved '{path}' to '{destination}'")

print("Receipt subtotal: $%.2f" % subtotal)

