import json
import os

students_dict = {

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


print (f"\ntype(students_dict): {type(students_dict)}")

filename = "students.txt"

# if needed clean-up  / rm filename
if os.path.exists(filename):
    os.remove(filename)
    print(f"File '{filename}' deleted successfully.")

print (f"\nWrite students_dict dict as str to {filename}")
try:
    with open(filename,'w') as data:
          data.write(str(students_dict))
except PermissionError:
    print (F"Error: file permission error occurred with writing {filename}")
else:
    print (f"\nRead {filename} contents")

print (f"\nRead and print {filename} contents")
try:
    with open(filename, 'r') as f:
        students_str = f.read()
        #for students_str in f:
        #    print(students_str)
except FileNotFoundError:
    print(f"Error: File not found: {filename}")
except IOError:
    print(f"Error: An error occurred while reading {filename}")
else:
    print(f"\nstudents_str (str from file):\n {students_str}\n")
    print (f"\ntype(students_str): {type(students_str)}\n")


print (f"Convert students_str to dict")
try:
    #students_dict = json.loads(students_str.replace("'", """))
    students_dict = eval(students_str)
#except json.JSONDecodeError:
except Exception:
    print(f"Error: Invalid format - student_str cannot be converted to students_dict")
else:
    print(f"\nstudents_dict (dict from str):\n {students_dict}\n")
    print (f"type(students_dict): {type(students_dict)}\n")

