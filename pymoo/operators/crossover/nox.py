from pymoo.core.crossover import Crossover
from pymoo.core.population import Population


class NoCrossover(Crossover):
    def __init__(self, *, n_parents=1, n_offsprings=None, prob=0.0, **kwargs):
        if n_offsprings is None:
            n_offsprings = n_parents
        super().__init__(n_parents, n_offsprings, prob, **kwargs)

    def do(self, problem, pop, *args, random_state, **kwargs):
        replace = self.n_offsprings > self.n_parents
        return Population.create(*[random_state.choice(parents, replace=replace) for parents in pop])
