
import os
import glob
import json
import shutil
import math

try:
    os.mkdir('./processed')
# if dir exists, it raises 'FileExistsError' exception which is in a subclass OSError exception
# The parent expection, OSError, will work with pre-3.3 version, but th FileExistError may not
except OSError:
    print("'./processed' directory already exists")


# get list of receipt files
subtotal = 0.0

# use glob.iglob() to return list of receipt files since it returns an iterator sequence [reduces memory footprint]
for path in glob.iglob('./new/receipt-[0-9]*.json'):
    with open(path) as f:
        # load receipt JSON contents to 'content' dict
        content = json.load(f)
        subtotal += float(content['value'])
        # str.replace() to create destination file name (change dir path from ./new/' to './processed/')
        destination = path.replace('new','processed')
        # move processed receipt files from './new' directory to './processed' dir
        shutil.move(path,destination)
        print(f"moved '{path}' to '{destination}'")

# use old python print formatting to round off subtotal
#print("Receipt subtotal: $%.2f" % subtotal)
# use round() built in function to round subtotal result to 2 decimal digits
print(f"Receipt subtotal: ${round(subtotal, 2)}")
# experiments with math.ceil() and math.floor() methods
#print(f"Receipt subtotal ceil:  ${math.ceil(subtotal)}")
#print(f"Receipt subtotal floor: ${math.floor(subtotal)}")


