class User:
    # class variable -  is a global variable to all "User" classes
    active_users = []

    def __init__(self, name, email):
        # instance variables
        self.name = name
        self.email = email

    def activate(self):
        #if self not in self.__class__.active_users:
        if not self.is_active():
            self.__class__.active_users.append(self)

    def deactivate(self):
        #if self not in self.__class__.active_users:
        if self.is_active():
            self.__class__.active_users.remove(self)

    def is_active(self):
        return self in self.__class__.active_users


# set 'name' instance variable, change it, then print it
me = User("Pat", "pat@example.com")
me.name = "Pat Hall"
print(me.name)

print (f"Active: {me.is_active()} Active Users: {User.active_users}")
me.activate()
me.activate()
print (f"Active: {me.is_active()} Active Users: {User.active_users}")
me.deactivate()
me.deactivate()
print (f"Active: {me.is_active()} Active Users: {User.active_users}")

print (f"Active users off of `me`: {me.active_users}, Class Level: {User.active_users}")
me.active_users = "Just me"
print (f"Active users off of `me`: {me.active_users}, Class Level: {User.active_users}")
