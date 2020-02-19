import numpy as np

from pymoo.model.individual import Individual
from pymoo.model.population import Population


class Initialization:

    def __init__(self,
                 sampling,
                 individual=Individual(),
                 repair=None,
                 eliminate_duplicates=None) -> None:

        super().__init__()
        self.sampling = sampling
        self.individual = individual
        self.repair = repair
        self.eliminate_duplicates = eliminate_duplicates

    def do(self, problem, n_samples, **kwargs):

        # provide a whole population object - (individuals might be already evaluated)
        if isinstance(self.sampling, Population):
            pop = self.sampling

        else:
            pop = Population(0, individual=self.individual)
            if isinstance(self.sampling, np.ndarray):
                pop = pop.new("X", self.sampling)
            else:
                pop = self.sampling.do(problem, n_samples, pop=pop, **kwargs)

        # repair all solutions that are not already evaluated
        if self.repair:
            I = [k for k in range(len(pop)) if pop[k].F is None]
            pop = self.repair.do(problem, pop[I], **kwargs)

        if self.eliminate_duplicates is not None:
            pop = self.eliminate_duplicates.do(pop)

        return pop
