from traceback import format_tb, format_stack

def my_func():
    raise IOError

def implicit_chaining():
    try:
        my_func()
    except IOError as err:
        1 / 0

def explicit_chaining():
    try:
        my_func()
    except IOError as err:
        raise ValueError("unexpected value") from err

# def nested():
#    try:
#        explicit_chaining()
#    except ValueError as err:
#        print()
#        import pdb; pdb.set_trace()
#        print()

def explicit_nested():
    try:
        explicit_chaining()
    except ValueError as err:
        traceback = format_tb(err.__traceback__)
        stacktrace = format_stack()
        with open("traceback.txt", "w") as f:
            f.write("\n".join(traceback))

        with open("stracktrace.txt", "w") as f:
            f.write("\n".join(stacktrace))

        print("ValueError caught here")




# implicit_chaining()
# explicit_chaining()
#nested()
explicit_nested()
