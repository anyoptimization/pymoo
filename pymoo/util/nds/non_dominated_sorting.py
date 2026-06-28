"""Non-dominated sorting by Pareto rank."""

import sys

import numpy as np
from moocore import pareto_rank, is_nondominated

from pymoo.util.dominator import Dominator


class NonDominatedSorting:
    def __init__(self, epsilon=None, method="fast_non_dominated_sort", dominator=None) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.method = method
        self.dominator = dominator

    def do(
        self,
        F,
        return_rank=False,
        only_non_dominated_front=False,
        n_stop_if_ranked=None,
        n_fronts=None,
        **kwargs,
    ):
        F = F.astype(float)

        if only_non_dominated_front:
            n_fronts = 1

        if len(F) == 0:
            fronts = []
        elif self.dominator is None:
            F_sort = F - self.epsilon if self.epsilon is not None else F
            ranks = pareto_rank(F_sort)
            fronts = []
            n_ranked = 0
            for r in range(ranks.max() + 1):
                if n_fronts is not None and r >= n_fronts:
                    break
                front = np.where(ranks == r)[0]
                fronts.append(front)
                n_ranked += len(front)
                if n_stop_if_ranked is not None and n_ranked >= n_stop_if_ranked:
                    break
        else:
            if n_stop_if_ranked is None:
                n_stop_if_ranked = int(1e8)
            if n_fronts is None:
                n_fronts = int(1e8)
            from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

            raw = fast_non_dominated_sort(F, dominator=self.dominator, **kwargs)
            fronts = []
            n_ranked = 0
            for front in raw:
                fronts.append(np.array(front, dtype=int))
                n_ranked += len(front)
                if n_ranked >= n_stop_if_ranked:
                    break

        if only_non_dominated_front:
            return fronts[0] if fronts else np.array([], dtype=int)

        if return_rank:
            rank = rank_from_fronts(fronts, F.shape[0])
            return fronts, rank

        return fronts


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, sys.maxsize, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank


# Returns all indices of F that are not dominated by the other objective values
def find_non_dominated(F, _F=None):
    if len(F) == 0:
        return np.array([], dtype=int)
    if _F is None:
        return np.where(is_nondominated(F.astype(float), keep_weakly=True))[0]
    else:
        M = Dominator.calc_domination_matrix(F, _F)
        return np.where(np.all(M >= 0, axis=1))[0]
