class MyMeta(type):
    def __new__(cls, name, bases, dct):
        final = super().__new__(cls, name, bases, dct)

        # modify class here
        return final

class MyClass(metaclass=MyMeta):
    pass

