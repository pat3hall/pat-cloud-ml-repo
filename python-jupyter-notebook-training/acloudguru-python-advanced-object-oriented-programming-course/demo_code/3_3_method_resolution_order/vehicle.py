class Vehicle:
    """
    Vehicle models a device tht can be used to travel.
    """
    def __init__(self, distance_traveled=0, unit='miles', **kwargs):
        print(f"__init__ from Vehicle with distance traveled: {distance_traveled} and {unit}")
        self.distance_traveled = distance_traveled
        self.unit = unit

    def description(self):
        return f"A {self.__class__.__name__} that has traveled {self.distance_traveled} {self.unit}"
