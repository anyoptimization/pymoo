from abc import abstractmethod


class Repair:
    """
    This class is allows to repair individuals after crossover if necessary.
    """

    def do(self, problem, pop, **kwargs):
        return self._do(problem, pop, **kwargs)

    @abstractmethod
    def _do(self, problem, pop, **kwargs):
        pass


class NoRepair(Repair):
    """
    A dummy class which can be used to simply do no repair.
    """

    def do(self, problem, pop, **kwargs):
        return pop