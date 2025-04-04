import sys

try:
    f = open('myfile.txt')
    s = f.readline()
    i = int(s.strip())
except ValueError:
    print("Could not convert data to an integer.")
except OSError as err:
    print(f"OS error:   err: {err}")
except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")
    raise
