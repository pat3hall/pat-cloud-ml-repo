from datetime import date

class MyInt(int):

    def __add__(self, other):
        return self.__class__(super().__add__(other))
    
    def __sub__(self, other):
            return self.__class__(super().__sub__(other))
    
    def days_from_now(self):
        # create an orderinal from today's date, add self, and then create the date from the new ordinal
        return date.fromordinal(date.today().toordinal() + self)