from abc import abstractmethod

import numpy as np


class Mutation:
    def do(self, problem, X, **kwargs):
        """

        Mutate variables in a genetic way.

        Parameters
        ----------
        problem : class
            The problem instance - specific information such as variable bounds might be needed.
        X : np.ndarray
            Two dimensional array where each row is an individual and each column represents a variable.

        Returns
        -------
        Y : np.ndarray
            The mutated individuals

        """
        Y = np.full(X.shape, np.inf)
        self._do(problem, X, Y, **kwargs)
        return Y

    @abstractmethod
    def _do(self, problem, X, **kwargs):
        pass
