from abc import abstractmethod, ABC

from pymoo.core.callback import Callback
from pymoo.core.problem import Problem


class DynamicProblem(Problem, ABC):
    pass


class DynamicTestProblem(DynamicProblem, ABC):

    def __init__(self, nt, taut, tau=1, time=None, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.nt = nt
        self.taut = taut
        self._time = time

    def tic(self, elapsed=1):

        # increase the time counter by one
        self.tau += elapsed

        # remove the cache of the problem to recreate ps and pf
        self.__dict__["cache"] = {}

    @property
    def time(self):
        if self._time is not None:
            return self._time
        else:
            return 1 / self.nt * (self.tau // self.taut)

    @time.setter
    def time(self, value):
        self._time = value


class TimeSimulation(Callback):

    def update(self, algorithm):
        problem = algorithm.problem
        if hasattr(problem, "tic"):
            problem.tic()
        else:
            raise Exception("TimeSimulation can only be used for dynamic test problems.")
