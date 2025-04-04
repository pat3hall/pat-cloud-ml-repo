from traceback import format_tb, format_stack

def my_func():
    raise IOError

def explicit_chaining():
    try:
        my_func()
    except IOError as err:
        raise ValueError("unexpected value") from err

explicit_chaining()
