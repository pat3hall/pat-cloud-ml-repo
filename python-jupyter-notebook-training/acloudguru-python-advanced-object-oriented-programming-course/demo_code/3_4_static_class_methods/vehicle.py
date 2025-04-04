from math import pi
class Vehicle:
    default_tire = 'tire'
    def __init__(self, engine, tires):
        self.engine = engine
        self.tires = tires

    @classmethod
    def bicycle(cls, tires=None):
        if not tires:
            tires = [cls.default_tire, cls.default_tire]
        return cls(None, tires)
    
    @staticmethod
    def wheel_circumference(radius):
        return 2 * pi * radius