import pickle

from datetime import date

class Person:
    def __init__(self, name: str, birthdate: date) -> None:
        self.name = name
        self.birthdate = birthdate

person1 = Person("Kevin Bacon", date(1958, 7, 8))
person2 = Person("Other Person", date.today())

# create list of persons:
people = [person1, person2]

# open pickle file in write byte mode and write list of Person objects to file
with open("people.pickle", "wb") as f:
    # pickle module has same API as the json module (e.g. dump, dumps, load, loads)
    # dump 'people' list of objects to a file
    pickle.dump(people, f)