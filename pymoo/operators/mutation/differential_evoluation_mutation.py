import numpy as np

from pymoo.model.mutation import Mutation
from pymoo.rand import random


class DifferentialEvolutionMutation(Mutation):
    def __init__(self, variant, CR):
        super().__init__()
        self.CR = CR
        self.variant = variant

    def _do(self, problem, pop, algorithm, **kwargs):

        X = pop.get("X")
        off = algorithm.off
        _X = off.get("X")

        # do the crossover
        if self.variant == "bin":
            # uniformly for each individual and each entry
            r = random.random(size=(len(off), problem.n_var)) < self.CR

        elif self.variant == "exp":

            # start point of crossover
            r = np.full((len(off), problem.n_var), False)

            # start point of crossover
            n = random.randint(0, problem.n_var, size=len(off))
            # length of chromosome to do the crossover
            L = random.random((len(off), problem.n_var)) < self.CR

            # create for each individual the crossover range
            for i in range(len(off)):
                # the actual index where we start
                start = n[i]
                for j in range(problem.n_var):

                    # the current position where we are pointing to
                    current = (start + j) % problem.n_var

                    # replace only if random value keeps being smaller than CR
                    if L[i, current]:
                        r[i, current] = True
                    else:
                        break

        else:
            raise Exception("Unknown crossover type. Either binomial or exponential.")

        X[r] = _X[r]
        return pop.new("X", X)
