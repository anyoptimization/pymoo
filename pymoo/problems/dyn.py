from abc import abstractmethod

from pymoo.core.problem import Problem


class DynamicProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = 1

    def next(self):

        # increase the time counter by one
        self.tau += 1

        # remove the cache of the problem to recreate ps and pf
        self.__dict__["cache"] = {}

    @abstractmethod
    def time(self):
        pass


class DynamicTestProblem(DynamicProblem):

    def __init__(self, nt, taut, **kwargs):
        super().__init__(**kwargs)
        self.nt = nt
        self.taut = taut

    def time(self):
        return 1 / self.nt * (self.tau // self.taut)
