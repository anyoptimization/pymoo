from abc import abstractmethod
from multimethod import overload, isa
from pymoo.model.population import Population, Individual


class Repair:
    """
    This class is allows to repair individuals after crossover if necessary.
    """

    @overload
    def do(self, problem, individual: isa(Individual), **kwargs):
        pop = Population.create(individual)
        return self.do(problem, pop, **kwargs)[0]

    @overload
    def do(self, problem, pop: isa(Population), **kwargs):
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