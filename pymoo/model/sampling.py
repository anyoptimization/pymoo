from abc import abstractmethod
import numpy as np

from pymop.problem import Problem


class Sampling:
    """

    This abstract class represents any sampling strategy that can be used
    to create an initial population or just an initial point

    """

    @abstractmethod
    def sample(self, problem, n_samples, data=None):
        """
        Sample new points according to the problem.


        Parameters
        ----------
        problem: class
            The problem to which points should be sampled. (lower and upper bounds, discrete, binary, ...)

        n_samples: int
            Number of points that should be created

        data: class
            Any additional data that might be necessary. e.g. constants of the algorithm, ...

        Returns
        -------

        """
        pass

    def sample_by_bounds(self, x_min, x_max, n_var, n_samples, data=None):
        class P(Problem):
            def __init__(self) -> None:
                self.n_var = n_var
                self.xl = np.full(n_var, x_min)
                self.xu = np.full(n_var, x_max)

        return self.sample(P(), n_samples, data)

