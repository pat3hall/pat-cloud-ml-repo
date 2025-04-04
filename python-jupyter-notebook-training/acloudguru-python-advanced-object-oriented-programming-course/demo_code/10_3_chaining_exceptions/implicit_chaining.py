# calling my_func will throw (raise) and IOError
def my_func():
    raise IOError

def implicit_chaining():
    try:
        my_func()
    except IOError as err:
        # this will cause a 2nd expection during the 'my_func()' IOError exception handling
        1 / 0

implicit_chaining()
