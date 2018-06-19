import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.rand import random


class DifferentialEvolutionCrossover(Crossover):

    def __init__(self, variant="DE/rand/1", prob=0.5, weight=0.8, method="binomial", bounce_back_in_bounds=True):
        super().__init__(3, 1)
        self.prob = prob
        self.weight = weight
        self.type = method
        self.variant = variant
        self.bounce_back_in_bounds = bounce_back_in_bounds

    def _do(self, p, parents, children, X=None, best_i=None):

        if X is None:
            raise Exception("Please provide the current population design space to do the crossover.")

        n_var = X.shape[1]

        # do the crossover
        if self.type == "binomial":

            # uniformly for each individual and each entry
            r = random.random(size=X.shape) < self.prob

        elif self.type == "exponential":

            r = np.full(X.shape, False)

            # start point of crossover
            n = random.randint(0, n_var, size=X.shape[0])
            # length of chromosome to do the crossover
            L = np.argmax((random.random(X.shape) > self.prob), axis=1)

            # create for each individual the crossover range
            for i in range(X.shape[0]):
                for l in range(L[i] + 1):
                    r[i, (n[i] + l) % n_var] = True

        else:
            raise Exception("Unknown crossover type. Either binomial or exponential.")

        # the so called donor vector
        children = np.copy(X)

        if self.variant == "DE/rand/1":
            trial = parents[:, 2] + self.weight * (parents[:, 0] - parents[:, 1])
        elif self.variant == "DE/target-to-best/1":
            trial = X + self.weight * (X[best_i, :] - X) + self.weight * (parents[:, 0] - parents[:, 1])
        elif self.variant == "DE/best/1":
            trial = X[best_i, :] + (parents[:, 0] - parents[:, 1]) * self.weight
        else:
            raise Exception("DE variant %s not known." % self.variant)

        # set only if larger than F_CR
        children[r] = trial[r]

        # bounce back into bounds
        if self.bounce_back_in_bounds:
            smaller_than_min, larger_than_max = p.xl > children, p.xu < children
            children[smaller_than_min] = (p.xl + (p.xl - children))[smaller_than_min]
            children[larger_than_max] = (p.xu - (children - p.xu))[larger_than_max]

        return children
