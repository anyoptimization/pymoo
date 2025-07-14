from pymoo.core.crossover import Crossover
from pymoo.core.population import Population


class NoCrossover(Crossover):
    def __init__(self, *, n_parents=1, n_offsprings=1, prob=0.0, **kwargs):
        super().__init__(n_parents, n_offsprings, prob, **kwargs)

    def do(self, problem, pop, *args, random_state, **kwargs):
        return Population.create(*[random_state.choice(parents) for parents in pop])
