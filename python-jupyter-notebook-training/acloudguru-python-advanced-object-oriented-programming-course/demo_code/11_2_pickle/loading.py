import pickle

from serial import Person

with open ("people.pickle", "rb") as f:
    people = pickle.load(f)

for person in people:
    print(type(person))
    print(person.__dict__)