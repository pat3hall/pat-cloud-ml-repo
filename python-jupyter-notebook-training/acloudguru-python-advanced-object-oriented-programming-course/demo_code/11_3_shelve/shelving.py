import shelve
from datetime import date
from serial import Person


person1 = Person("Kevin Bacon", date(1958, 7, 8))
person2 = Person("Other Person", date.today())

# DB_NAME is the file name with flag='c' (Open database for reading and writing, creating it if it doesn’t exist)
DB_NAME = "people.shelf"
with shelve.open(DB_NAME, "c") as shelf:
    shelf["kevin"] = person1
    shelf["other"] = person2
    shelf["count"] = 2

# re-open DB_NAME file, this time with flag='r' (Open existing database for reading only)
with shelve.open(DB_NAME, "r") as shelf:
    print(f"Keys: {list(shelf.keys())}")
    print(f"Values: {list(shelf.values())}")
    
    print(f'\ntype(shelf["kevin"]):        {type(shelf["kevin"])}')
    print(f'shelf["kevin"].__dict__:     {shelf["kevin"].__dict__}')
    print(f'shelf["other"].__dict__:     {shelf["other"].__dict__}')