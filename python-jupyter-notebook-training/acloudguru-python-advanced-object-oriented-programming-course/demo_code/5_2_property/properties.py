class Employee:
    def __init__(self, name, salary, hours_per_week=40) -> None:
        self.name = name
        self._hours_per_week = hours_per_week
        self._salary = salary
        self.__set_hourly_rate()
    
    @property
    def hours_per_week(self):
        return self._hours_per_week
    
    @hours_per_week.setter
    def hours_per_week(self, value):
        self._hours_per_week = value
        self.__set_hourly_rate()
        
    @hours_per_week.deleter
    def hours_per_week(self):
        del self._hours_per_week
        
    @property
    def salary(self):
        return self._salary
    
    @salary.setter
    def salary(self, value):
        self._salary = value
        self.__set_hourly_rate()
        
    @salary.deleter
    def salary(self):
        del self._salary

    @property
    def hourly_rate(self):
        return self._hourly_rate
        

    # do not want hours_per_week, hour_rate, etc. to be publicly accessible
    def __set_hourly_rate(self):
        self._hourly_rate = round(self._salary / 52 / self._hours_per_week, 2)


if __name__ == '__main__':
    e = Employee('Kevin', 80000)
    print(f"Salary: {e.salary}")
    print(f"Hours Per Week: {e.hours_per_week}")
    print(f"Hourly Rate: {e.hourly_rate}")

    print(f"\nChanging salary\n")
    e.salary = 90000

    print(f"Salary: {e.salary}")
    print(f"Hours Per Week: {e.hours_per_week}")
    print(f"Hourly Rate: {e.hourly_rate}")

    print(f"\nChanging hours per week\n")
    e.hours_per_week = 35

    print(f"Salary: {e.salary}")
    print(f"Hours Per Week: {e.hours_per_week}")
    print(f"Hourly Rate: {e.hourly_rate}")

