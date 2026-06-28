"""Dynamic optimization problem base classes."""

from abc import ABC

from pymoo.core.callback import Callback
from pymoo.core.problem import Problem


class DynamicProblem(Problem, ABC):
    """Base class for dynamic optimization problems."""

    pass


class DynamicTestProblem(DynamicProblem, ABC):
    """Base class for dynamic test problems with time-dependent behavior.

    Args:
        nt: Number of time instances.
        taut: Time unit of change.
        tau: Initial time value.
        time: Optional custom time function.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, nt: int, taut: int, tau: int = 1, time=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tau = tau
        self.nt = nt
        self.taut = taut
        self._time = time

    def tic(self, elapsed: int = 1) -> None:
        """Increment the time counter and clear caches.

        Args:
            elapsed: Time increment.
        """
        # increase the time counter by one
        self.tau += elapsed

        # remove the cache of the problem to recreate ps and pf
        self.__dict__["cache"] = {}

    @property
    def time(self):
        """Get the current time value.

        Returns:
            Current time value.
        """
        if self._time is not None:
            return self._time
        else:
            return 1 / self.nt * (self.tau // self.taut)

    @time.setter
    def time(self, value):
        """Set a custom time value.

        Args:
            value: Custom time value.
        """
        self._time = value


class TimeSimulation(Callback):
    """Callback to update time in dynamic problems during optimization."""

    def update(self, algorithm):
        """Update time counter in the problem.

        Args:
            algorithm: The optimization algorithm.

        Raises:
            Exception: If problem does not have tic method.
        """
        problem = algorithm.problem
        if hasattr(problem, "tic"):
            problem.tic()
        else:
            raise Exception("TimeSimulation can only be used for dynamic test problems.")
