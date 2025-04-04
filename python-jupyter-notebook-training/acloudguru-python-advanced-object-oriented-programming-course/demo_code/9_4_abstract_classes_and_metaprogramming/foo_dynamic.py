# Creating a Class Dynamically
#   - 'type' constructer can handle 3 args, a 'name', 'base classes', and 'attributes' (passed in via a 'dict')

def foo_init(self, name):
    self.name = name

def foo_print_name(self):
    print(self.name)

Foo = type("Foo", (), {
    "__init__" : foo_init,
    "print_name" : foo_print_name,
})


foo_fred = Foo("Fred")
print (f"foo_fred.name: {foo_fred.name}")
print (f"foo_fred.print_name(): ")
foo_fred.print_name()
