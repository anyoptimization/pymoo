import numpy as np

from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function


class NonDominatedSorting:

    def __init__(self, epsilon=None, method="fast_non_dominated_sort") -> None:
        super().__init__()
        self.epsilon = epsilon
        self.method = method

    def do(self, F, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None, **kwargs):
        F = F.astype(float)

        # if not set just set it to a very large values because the cython algorithms do not take None
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)
        func = load_function(self.method)

        # set the epsilon if it should be set
        if self.epsilon is not None:
            kwargs["epsilon"] = float(self.epsilon)

        fronts = func(F, **kwargs)

        # convert to numpy array for each front and filter by n_stop_if_ranked if desired
        _fronts = []
        n_ranked = 0
        for front in fronts:

            _fronts.append(np.array(front, dtype=int))

            # increment the n_ranked solution counter
            n_ranked += len(front)

            # stop if more than this solutions are n_ranked
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
    rank = np.full(n, 1e16, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank


# Returns all indices of F that are not dominated by the other objective values
def find_non_dominated(F, _F=None):
    M = Dominator.calc_domination_matrix(F, _F)
    I = np.where(np.all(M >= 0, axis=1))[0]
    return I
