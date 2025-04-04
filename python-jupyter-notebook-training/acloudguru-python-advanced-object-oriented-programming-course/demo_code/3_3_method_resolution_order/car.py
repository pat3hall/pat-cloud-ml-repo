from vehicle import Vehicle

class Car(Vehicle):
    default_tire = 'tire'

    def __init__(self, engine, tires=[],distance_traveled=0, unit='miles', **kwargs):
        print(f"__init__ from Car with distance traveled: {distance_traveled} and {unit}")
        super().__init__(distance_traveled=distance_traveled, unit=unit, **kwargs)
        if not tires:
            tires = [self.default_tire, self.default_tire]
        self.tires = tires
        self.engine = engine

    def drive(self, distance):
        self.distance_traveled += distance

