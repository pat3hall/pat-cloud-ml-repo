from traceback import format_tb, format_stack

def my_func():
    raise IOError

def explicit_chaining():
    try:
        my_func()
    except IOError as err:
        raise ValueError("unexpected value") from err

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

explicit_nested()
