import abc

import numpy as np


class Indicator:

    def __init__(self):
        self.default_if_empty = 0.0

    def calc(self, F):

        # if it is a 1d array
        if len(F.shape) == 1:
            F = np.array([F])

        # if there are not data
        if F.shape[1] == 0:
            return self.default_if_empty

        return self._calc(F)

    @abc.abstractmethod
    def _calc(self, F):
        return


