four_sided = {
    "name": "A 4 sided die",
    "options": [1, 2, 3, 4]
}

def changeDie(die):
    die["name"] = "A 6 sided die"
    die["options"].append(5)
    die["options"].append(6)
    return die

print("\nUsing changeDie without copy\n")
new_die = changeDie(four_sided)
print(f"four_sided: {four_sided}")
print(f"new_die:    {new_die}")
print(f"Id of four_sided is {id(four_sided)}")
print(f"Id of new_die    is {id(new_die)}")

import copy
four_sided = {
    "name": "A 4 sided die",
    "options": [1, 2, 3, 4]
}

def changeDie(die):
    die = copy.copy(die)
    die["name"] = "A 6 sided die"
    die["options"].append(5)
    die["options"].append(6)
    return die
print("\nUsing changeDie with copy\n")
new_die = changeDie(four_sided)
print(f"four_sided: {four_sided}")
print(f"new_die:    {new_die}")
print(f"Id of four_sided is {id(four_sided)}")
print(f"Id of new_die    is {id(new_die)}")


four_sided = {
    "name": "A 4 sided die",
    "options": [1, 2, 3, 4]
}

def changeDie(die):
    die = copy.deepcopy(die)
    die["name"] = "A 6 sided die"
    die["options"].append(5)
    die["options"].append(6)
    return die
print("\nUsing changeDie with deepcopy\n")
new_die = changeDie(four_sided)
print(f"four_sided: {four_sided}")
print(f"new_die:    {new_die}")
print(f"Id of four_sided is {id(four_sided)}")
print(f"Id of new_die    is {id(new_die)}")


print("\nUsing slicing to create a unique new list")
my_list = [1, 2, 3, 4]
my_other_list = my_list[:]
print(f"my_list       {my_list}")
print(f"my_other_list {my_other_list}")
print(f"my_other_list == my_list\n{my_other_list == my_list}")
print(f"my_other_list is my_list\n{my_other_list is my_list}")
