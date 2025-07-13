from abc import abstractmethod

from pymoo.core.operator import Operator
from pymoo.core.population import Population
from pymoo.util import default_random_state


class Sampling(Operator):

    def __init__(self) -> None:
        """
        This abstract class represents any sampling strategy that can be used to create an initial population or
        an initial search point.
        """
        super().__init__()

    @default_random_state
    def do(self, problem, n_samples, *args, random_state=None, **kwargs):
        """
        Sample new points with problem information if necessary.

        Parameters
        ----------

        problem : :class:`~pymoo.core.problem.Problem`
            The problem to which points should be sampled. (lower and upper bounds, discrete, binary, ...)

        n_samples : int
            Number of samples

        Returns
        -------
        pop : Population
            The output population after sampling

        """
        val = self._do(problem, n_samples, *args, random_state=random_state, **kwargs)
        return Population.new("X", val)

    @abstractmethod
    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        pass



