import random

class MeasurementGen():
    def __init__(self):
        super().__init__()
        self.measurements = 0

    def get_measurements(self,i):
        # productions, consumptions, tariffs
        return random.sample(range(10), 4)
