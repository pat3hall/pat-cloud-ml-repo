
def my_func():
    raise IOError

def explicit_chaining():
    try:
        my_func()
    except IOError as err:
        raise ValueError("unexpected value") from err

def nested():
   try:
       explicit_chaining()
   except ValueError as err:
       print()
       import pdb; pdb.set_trace()
       print()

nested()
