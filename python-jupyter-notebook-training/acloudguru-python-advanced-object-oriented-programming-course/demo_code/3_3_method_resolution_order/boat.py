from vehicle import Vehicle

class Boat(Vehicle):

    def __init__(self, boat_type='sail',distance_traveled=0, unit='miles', **kwargs):
        print(f"__init__ from Boat with distance traveled: {distance_traveled} and {unit}")
        super().__init__(distance_traveled=distance_traveled, unit=unit, **kwargs)
        self.boat_type = boat_type

    def voyage(self, distance):
        self.distance_traveled += distance

    def description(self):
        initial = super().description()
        return f"{initial} using a {self.boat_type}"
