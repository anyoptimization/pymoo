from abc import abstractmethod


class Repair:
    """
    This class is allows to repair individuals after crossover if necessary.
    """

    def do(self, pop, problem, **kwargs):
        return self._do(pop, problem, **kwargs)

    @abstractmethod
    def _do(self, pop, problem, **kwargs):
        pass
