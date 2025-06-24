import sys

import numpy as np

from pymoo.functions import load_function
from pymoo.util.dominator import Dominator


class NonDominatedSorting:

    def __init__(self, epsilon=None, method="fast_non_dominated_sort", dominator=None) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.method = method
        self.dominator = dominator

    def do(self, F, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None, n_fronts=None, **kwargs):
        F = F.astype(float)

        # if not set just set it to a very large values because the cython algorithms do not take None
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)

        # if only_non_dominated_front is True, we only need 1 front
        if only_non_dominated_front:
            n_fronts = 1
        elif n_fronts is None:
            n_fronts = int(1e8)

        # if a custom dominator is provided, use the custom dominator and run fast_non_dominated_sort
        if self.dominator is not None:
            # Use the custom dominator directly
            from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
            fronts = fast_non_dominated_sort(F, dominator=self.dominator, **kwargs)
        else:
            # Use the standard function loader approach
            func = load_function(self.method)

            # set the epsilon if it should be set
            if self.epsilon is not None:
                kwargs["epsilon"] = float(self.epsilon)

            # add n_fronts parameter if the method supports it
            if self.method == "fast_non_dominated_sort":
                kwargs["n_fronts"] = n_fronts
                kwargs["n_stop_if_ranked"] = n_stop_if_ranked

            fronts = func(F, **kwargs)

        # convert to numpy array for each front and filter by n_stop_if_ranked
        _fronts = []
        n_ranked = 0
        for front in fronts:

            _fronts.append(np.array(front, dtype=int))

            # increment the n_ranked solution counter
            n_ranked += len(front)

            # stop if more solutions than n_ranked are ranked
            if n_ranked >= n_stop_if_ranked:
                break

        fronts = _fronts

        if only_non_dominated_front:
            return fronts[0]

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
def find_non_dominated(F, _F=None, func=load_function("find_non_dominated")):
    if _F is None:
        indices = func(F.astype(float))
        return np.array(indices, dtype=int)
    else:
        # Fallback to the matrix-based approach when _F is provided
        M = Dominator.calc_domination_matrix(F, _F)
        I = np.where(np.all(M >= 0, axis=1))[0]
        return I
