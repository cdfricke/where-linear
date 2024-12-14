import numpy as np

class DomainFinder:
    """
    The goal of this class is to act as a simple interface to the
    algorithms available to find a linear domain within the data.
    """
    def __init__(self):
        self.domain = (0.0, 0.0)    # the discovered domain (horizontal)
        self.range = (0.0, 0.0)     # discovered range (vertical)
        self.slope = 0.0            # the resulting slope of the fit
        self.intercept = 0.0        # the resulting y-intercept of the fit
        self._xdata = np.zeros(1)   # values of points along x-axis
        self._ydata = np.zeros(1)   # values of points along y-axis

    def setXData(self, data: list):
        self._xdata = np.array(data)
    
    def setYData(self, data: list):
        self._ydata = np.array(data)


    