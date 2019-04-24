import numpy as np

from pymoo.model.repair import Repair


class BoundsBackRepair(Repair):

    def _do(self, problem, pop, **kwargs):

        # bring back to bounds if violated through crossover - bounce back strategy
        X = pop.get("X")
        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)

        # otherwise bounds back into the feasible space
        _range = xu - xl
        X[X < xl] = (xl + np.mod((xl - X), _range))[X < xl]
        X[X > xu] = (xu - np.mod((X - xu), _range))[X > xu]

        return pop.new("X", X)
