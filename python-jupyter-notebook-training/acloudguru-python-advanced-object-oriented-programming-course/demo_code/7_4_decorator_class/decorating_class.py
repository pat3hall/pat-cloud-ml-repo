def print_something(func):
    def wrapper_func(*args, **kwargs):
        print (f"We're running {func}")
        return func(*args, **kwargs)
    return wrapper_func

@print_something
class Example:
    def __init__(self, name):
        self.name = name

example = Example("Kevin Bacon")
print(example)


from dataclasses import dataclass
from datetime import date

@dataclass
class Record:
    name: str
    artist_name: str
    release_date: date

record1 = Record("The Beaatles", "Let it be", date(1970, 5, 8))
record2 = Record("The Beaatles", "Let it be", date(1970, 5, 8))
print("\n")
print(record1.name)
print(record1.artist_name)
print(record1.release_date)
print(record1 == record2)

# has optional 'func' function input, and '*" means it does not allow positional argument 
def log_to_file(func = None, *, file_name='default_file.txt'):
    def decorator_func(func):
        def wrapper_func(*args, **kwargs):
            with open(file_name, "a") as f:
                output = func.__name__ + \
                    f"called with Args: {args}" + \
                    f" called with Kwargs: {kwargs}\n"
                f.write(output)
            return func(*args, **kwargs)
        return wrapper_func
    if func:
        return decorator_func(func)
    else:
        return decorator_func

            
        

