from abc import abstractmethod

import numpy

from pymoo.model import random
from pymoo.model.evaluator import Evaluator
from pymoo.util.non_dominated_rank import NonDominatedRank


class Algorithm:

    def solve(self, problem, evaluator, seed=1, return_only_feasible=True, return_only_non_dominated=True):

        # set the random seed
        random.seed(seed)

        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(evaluator)

        # call the algorithm to solve the problem
        x, f, g = self._solve(problem, evaluator)

        if return_only_feasible:
            if g is not None and g.shape[0] == len(f) and g.shape[1] > 0:
                b = numpy.array(numpy.where(numpy.sum(g, axis=1) <= 0))[0]
                x = x[b, :]
                f = f[b, :]
                if g is not None:
                    g = g[b, :]

        if return_only_non_dominated:
            idx_non_dom = NonDominatedRank.calc_as_fronts(f,g)[0]
            x = x[idx_non_dom, :]
            f = f[idx_non_dom, :]
            if g is not None:
                g = g[idx_non_dom, :]

        return x, f, g

    @abstractmethod
    def _solve(self, problem, evaluator):
        pass
