import abc

from pymoo.model.population import Population
from pymoo.model.repair import Repair
import numpy as np


def is_out_of_bounds(X, xl, xu):
    return np.where(np.any(np.logical_or(X < xl, X > xu), axis=1))[0]


def is_out_of_bounds_by_problem(problem, X):
    return is_out_of_bounds(X, problem.xl, problem.xu)


class BoundsRepair(Repair):

    def _do(self,
            problem,
            pop_or_X,
            check_out_of_bounds=True,
            **kwargs):

        is_array = not isinstance(pop_or_X, Population)

        X = pop_or_X if is_array else pop_or_X.get("X")

        X = self.repair_out_of_bounds(problem, X, **kwargs)

        assert len(is_out_of_bounds_by_problem(problem, X)) == 0

        if is_array:
            return X
        else:
            pop_or_X.set("X", X)
            return pop_or_X

    @abc.abstractmethod
    def repair_out_of_bounds(self, problem, X, **kwargs):
        pass
