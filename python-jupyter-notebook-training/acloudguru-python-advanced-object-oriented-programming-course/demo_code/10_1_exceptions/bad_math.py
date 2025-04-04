#import traceback

def bad_math():
    try:
        10 / 0
    except ZeroDivisionError as err:
        # access traceback as a string while still handling the exception
        tb = traceback.format_tb(err.__traceback__)
        print(tb)

