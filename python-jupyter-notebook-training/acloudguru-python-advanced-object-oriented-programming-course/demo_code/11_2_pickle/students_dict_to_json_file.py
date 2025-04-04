import json
import os

students = {

  'Student 1': {
        'Name': "Alice", 'Age' :10, 'Grade':4,
    },
    'Student 2': {
        'Name':'Bob', 'Age':11, 'Grade':5
    },
    'Student 3': {
        'Name':'Elena', 'Age':14, 'Grade':8
    }
}


print (f"\ntype(students): {type(students)}")

json_file = "students.json"

# if needed clean-up  / rm json_file
if os.path.exists(json_file):
    os.remove(json_file)
    print(f"File '{json_file}' deleted successfully.")

print (f"\nWrite students dict as json to {json_file}")
try:
    with open(json_file, 'w') as file:
      json.dump(students, file, indent=4,  sort_keys=True, separators=(',',':'))
except PermissionError:
    print (F"Error: file permission error occurred with writing {json_file}")
else:
    print (f"\nRead {json_file} contents")


print (f"\nRead json from {json_file} to students dict")
try:
    with open(json_file, 'r') as file:
        students = json.load(file)
except FileNotFoundError:
    print(f"Error: File not found: {json_file}")
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in: {json_file}")
except IOError:
    print(f"Error: An error occurred while reading {json_file}")

print(f"\nstudents (dict from json file):\n {students}\n")

print (f"type(students): {type(students)}\n")

