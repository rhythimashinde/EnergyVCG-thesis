class BaseMeasurementGen():
    def __init__(self):
        raise NotImplementedError("Please Implement this method")

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        raise NotImplementedError("Please Implement this method")
