import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.rand import random


class DifferentialEvolutionCrossover(Crossover):

    def __init__(self, variant="DE/rand/1", prob=0.5, weight=0.8, method="binomial", bounce_back_in_bounds=True):
        super().__init__(4, 1)
        self.prob = prob
        self.weight = weight
        self.type = method
        self.variant = variant
        self.bounce_back_in_bounds = bounce_back_in_bounds

    def _do(self, p, parents, children, **kwargs):

        n_var = parents.shape[2]
        n_offsprings = parents.shape[0]

        # do the crossover
        if self.type == "binomial":

            # uniformly for each individual and each entry
            r = random.random(size=(n_offsprings, n_var)) < self.prob

        elif self.type == "exponential":

            r = np.full((n_offsprings, n_var), False)

            # start point of crossover
            n = random.randint(0, n_var, size=n_var)
            # length of chromosome to do the crossover
            L = np.argmax((random.random(n_offsprings) > self.prob), axis=1)

            # create for each individual the crossover range
            for i in range(n_offsprings):
                for l in range(L[i] + 1):
                    r[i, (n[i] + l) % n_var] = True

        else:
            raise Exception("Unknown crossover type. Either binomial or exponential.")

        # the so called donor vector
        children[:, :] = parents[:, 0]

        if self.variant == "DE/rand/1":
            trial = parents[:, 1] + self.weight * (parents[:, 2] - parents[:, 3])
        else:
            raise Exception("DE variant %s not known." % self.variant)

        # set only if larger than F_CR
        children[r] = trial[r]

        # bounce back into bounds
        if self.bounce_back_in_bounds:
            smaller_than_min, larger_than_max = p.xl > children, p.xu < children
            children[smaller_than_min] = (p.xl + (p.xl - children))[smaller_than_min]
            children[larger_than_max] = (p.xu - (children - p.xu))[larger_than_max]
