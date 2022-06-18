import numpy as np

from pymoo.core.operator import Operator


class Repair(Operator):

    def do(self, problem, pop, **kwargs):
        X = np.array([ind.X for ind in pop])
        if self.vtype is not None:
            X = X.astype(self.vtype)

        Xp = self._do(problem, X, **kwargs)

        pop.set("X", Xp)
        return pop

    def _do(self, problem, X, **kwargs):
        return X


class NoRepair(Repair):
    pass
