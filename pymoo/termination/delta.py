import math
from abc import abstractmethod

from pymoo.core.termination import Termination


class DeltaToleranceTermination(Termination):

    def __init__(self, tol, n_skip=0):
        super().__init__()

        # the tolerance threshold the difference (delta) to be under
        assert tol >= 0
        self.tol = tol

        # the previous values to calculate the difference
        self.data = None

        # a counter of update calls
        self.counter = 0

        # whether some updates should be skipped
        self.n_skip = n_skip

    def _update(self, algorithm):

        # the object from the previous iteration
        prev = self.data

        # and the one from the current iteration
        current = self._data(algorithm)

        # if there is no previous element to use
        if prev is None:
            perc = 0.0

        elif self.counter > 0 and self.counter % (self.n_skip + 1) != 0:
            perc = self.perc

        else:
            tol = self.tol
            delta = self._delta(prev, current)

            if delta <= tol:
                return 1.0
            else:
                v = (delta - tol)
                perc = 1 / (1 + v)

        # remember the data from the current iteration and set it to data
        self.data = current

        # increase the function update counter
        self.counter += 1

        return perc

    @abstractmethod
    def _delta(self, prev, current):
        pass

    @abstractmethod
    def _data(self, algorithm):
        pass
