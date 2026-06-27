"""Sampling operators for population initialization."""

from abc import abstractmethod

from pymoo.core.operator import Operator
from pymoo.core.population import Population
from pymoo.util import default_random_state


class Sampling(Operator):
    def __init__(self) -> None:
        """Initialize a Sampling operator.

        This abstract class represents any sampling strategy that can be used
        to create an initial population or an initial search point.
        """
        super().__init__()

    @default_random_state
    def do(self, problem, n_samples, *args, random_state=None, **kwargs):
        """Sample new points with problem information if necessary.

        Args:
            problem: The problem to which points should be sampled
                (lower and upper bounds, discrete, binary, ...).
            n_samples: Number of samples.
            *args: Additional positional arguments.
            random_state: Random state for reproducibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Population: The output population after sampling.
        """
        val = self._do(problem, n_samples, *args, random_state=random_state, **kwargs)
        return Population.new("X", val)

    @abstractmethod
    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        pass
