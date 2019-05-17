import numpy as np

from pymoo.model.repair import Repair


class OutOfBoundsRepair(Repair):

    def _do(self, problem, pop, **kwargs):

        X = pop.get("X")
        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)

        X[X < xl] = xl[X < xl]
        X[X > xu] = xu[X > xu]

        pop.set("X", X)
        return pop
