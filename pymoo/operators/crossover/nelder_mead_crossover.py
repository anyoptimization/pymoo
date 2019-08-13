import numpy as np

from pymoo.algorithms.so_nelder_mead import max_expansion_factor
from pymoo.model.crossover import Crossover


class NelderMeadCrossover(Crossover):

    def __init__(self, problem, **kwargs):
        super().__init__(problem.n_var + 1, 1, **kwargs)

    def do(self, problem, pop, parents, **kwargs):
        n = problem.n_var - 1

        # obtain the matrices from the population
        X = pop.get("X")[parents]
        F = pop.get("F")[parents][..., 0]

        # the array to fill
        _X = np.full((len(X), problem.n_var), np.nan)

        for k, P in enumerate(parents):
            # sort the x values by their corresponding F values
            x = X[k, :][np.argsort(F[k])]

            # calculate the centroid of n best points
            centroid = x[:n + 1].mean(axis=0)

            # calculate the vector from the worst to the centroid
            v = centroid - x[n + 1]

            # maximum factor until the boundaries are hit
            max_factor = max_expansion_factor(centroid, v, problem)

            # randomly chose the extension, expansion or contraction through a factor
            factor = np.random.random() * min(3, max_factor)

            # calculate the offspring
            _X[k] = x[n + 1] + factor * v

        return pop.new("X", _X)
