def double(func):
    def wrapper_func(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2
    return wrapper_func

def add(a, b):
    return a + b

@double
def xadd(a, b):
    return a + b

print(add(1, 2))
print(xadd(1,2))