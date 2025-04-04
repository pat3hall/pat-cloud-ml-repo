def log(func=None, *, file_name=None):
    def decorator_func(func):
        def wrapped_func(*args, **kwargs):

            result = func(*args, **kwargs)

            message = f"running: {func.__name__} args: {args} kwargs: {kwargs}, result: {result}"

            if file_name:
                with open(file_name, 'a') as f:
                    f.write(message + "\n")

            else:
                print(message)

            return result

        return wrapped_func
    if func:
        return decorator_func(func)
    else:
        return decorator_func

from datetime import datetime

def benchmark(func=None, *, file_name=None):
    def decorator_func(func):
        def wrapped_func(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = round((end_time - start_time).total_seconds(), 1)
            message = f"benchmark: {func.__name__} duration: {duration}"

            if file_name:
                with open(file_name, 'a') as f:
                    f.write(message + "\n")

            else:
                print(message)

            return result

        return wrapped_func
    if func:
        return decorator_func(func)
    else:
        return decorator_func


if __name__ == '__main__':
    import time
    import os
    print("\nTesting log decorator with 'add' function")
    @log
    def add(a, b):
        return a + b

    add(1, b=2)

    print("\nTesting log decorator with 'add' function and output file")
    file_name="log.txt"
    @log(file_name=file_name)
    def add(a, b):
        return a + b

    add(1, b=2)
    print(f"Printing '{file_name}' contents, then deleting it")
    if os.path.exists(file_name):
        print(open(file_name).read())
        os.remove(file_name)
    else:
        print(f"Error: {file_name} does not exist")


    print("\nTesting benchmark decorator with 'add' function")
    @benchmark
    def add(a, b):
        time.sleep(2)
        return a + b

    add(1, b=2)

    print("\nTesting benchmark decorator with 'add' function and output file")
    file_name="benchmark.txt"
    @benchmark(file_name=file_name)
    def add(a, b):
        time.sleep(2)
        return a + b

    add(1, b=2)
    print(f"Printing '{file_name}' contents, then deleting it")
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            print(f.read())
        os.remove(file_name)
    else:
        print(f"Error: {file_name} does not exist")
