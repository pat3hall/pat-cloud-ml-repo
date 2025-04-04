import datetime
from datetime import date
#import dataclasses

# New DataClass
class NewDataClass(type):
    def __new__(cls, name, bases, dct):
        final = super().__new__(cls, name, bases, dct)
        # pdb - The Python Debugger ; set_trace() adds a breakpoint
        # import pdb; pdb.set_trace()

        defaults = {
            key: value
            for key,value in final.__dict__.items()
            if not key.startswith("_")
        }

        #for key,value in final.__dict__.items():
        #    print(f"key: {key}, value: {value}")

        print(f"\ndefaults: {defaults}\n")

        init = final.__create_init(final.__annotations__, defaults)

        def eq(self, other):
            return self.__dict__ == other.__dict__

        def ne(self, other):
            return self.__dict__ != other.__dict__

        setattr(final, "__init__", init)
        setattr(final, "__eq__", eq)
        setattr(final, "__ne__", ne)

        return final

    @staticmethod
    def __create_init(annotations, defaults, *, return_type=None):
        name = "__init__"
        args = ["self"]
        default_args = []
        body_lines = []
        for key,value in annotations.items():
            print(f"annotations.items(): key: {key}, value: {value}, value.__name__: {value.__name__}")
            if key in defaults.keys():
                default_args.append(f"{key}:{value.__name__}={repr(defaults[key])}")
                print(f"\ndefault:  {key}:{value.__name__}={repr(defaults[key])}\n")
            else:
                args.append(f"{key}:{value.__name__}")
                # print(f"{key}:{value.__name__}")
            body_lines.append(f"self.{key} = {key}")

        args = ", ".join(args + default_args)
        body = "\n ".join(body_lines)

        text = f"def {name}({args})->{return_type}:\n {body}"
        # create function with 'exec()'
        # print("Debug 'text'")
        # print(text)
        # print("\n")
        exec(text)
        return locals()[name]

class Record(metaclass=NewDataClass):
    # alternative for setting metaclass:
    # __metaclass__ = NewDataClass
    artist: str
    title: str
    release_date: datetime.date = datetime.date.today()
    album_color: str = "blue"

# Note: print of 'record*' does not work because __repr__ was not implemented in NewDataClass
record1 = Record("The Beatles", "With the Beatles", datetime.date(1963, 11, 22), "red")
#print (f"\nrecord1:  {record1}")
record2 = Record("The Beatles", "With the Beatles", datetime.date(1963, 11, 22), "red")
#print (f"record2:  {record2}")
record3 = Record("The Beatles", "Help", datetime.date(1963, 11, 22))
#print (f"record3:  {record3}")

print (f"\nrecord1.artist:       {record1.artist}")
assert record1.artist == "The Beatles"
print (f"record1.title:        {record1.title}")
assert record1.title == "With the Beatles"
print (f"record1.album_color:  {record1.album_color}")
assert record1.album_color == "red"
print (f"record1.release_date: {record1.release_date}")
assert record1.release_date == datetime.date(1963, 11, 22)

# was not able to get below assert to work because it set "release_date: date: datetime.date.today()"
#   instead of  "release_date: datetime.date: datetime.date.today()"
#assert record3.release_date == datetime.date.today()

print (f"\nrecord3.album_color:  {record3.album_color}")
assert record3.album_color == "blue"
assert record1 == record2
assert record1 != record3

