from pymoo.core.crossover import Crossover
from pymoo.core.population import Population


class NoCrossover(Crossover):
    def __init__(self, *, n_parents=1, n_offsprings=None, prob=0.0, **kwargs):
        if n_offsprings is None:
            n_offsprings = n_parents
        super().__init__(n_parents, n_offsprings, prob, **kwargs)

    def do(self, problem, pop, *args, random_state, **kwargs):
        offsprings = []
        for parents in pop:
            if self.n_offsprings < self.n_parents:
                # Select without replacement
                offsprings.extend(random_state.choice(parents, size=self.n_offsprings, replace=False))
            elif self.n_offsprings == self.n_parents:
                # Return all parents as-is
                offsprings.extend(parents)
            else:
                # Keep each parent at least once, then fill randomly
                offsprings.extend(parents)
                extra = self.n_offsprings - self.n_parents
                offsprings.extend(random_state.choice(parents, size=extra, replace=True))
        return Population.create(*offsprings)
