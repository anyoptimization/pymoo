from abc import abstractmethod

import numpy as np


class Mutation:
    def do(self, problem, X):
        Y = np.full(X.shape, np.inf)
        return self._do(problem, X, Y)

    @abstractmethod
    def _do(self, problem, X):
        pass
