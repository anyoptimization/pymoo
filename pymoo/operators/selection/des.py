import numpy as np

from pymoo.core.selection import Selection


class DES(Selection):
    """DE-specific parent selection that returns index matrices for each variant."""

    def __init__(self, variant, **kwargs):
        super().__init__(**kwargs)
        self.variant = variant

    def _do(self, problem, pop, n_select, n_parents, random_state=None, **kwargs):
        variant = self.variant

        if variant == "ranked":
            P = self._rand(pop, n_select, n_parents, random_state)
            P[:, 1:] = _rank_sort(P[:, 1:], pop)
        elif variant == "best":
            P = self._best(pop, n_select, n_parents, random_state)
        elif variant == "current-to-best":
            P = self._current_to_best(pop, n_select, n_parents, random_state)
        elif variant == "current-to-rand":
            P = self._current_to_rand(pop, n_select, n_parents, random_state)
        else:
            P = self._rand(pop, n_select, n_parents, random_state)

        return P

    def _rand(self, pop, n_select, n_parents, random_state):
        n_pop = len(pop)
        P = np.empty([n_select, n_parents], dtype=int)
        P[:, 0] = np.arange(n_pop)
        for j in range(1, n_parents):
            P[:, j] = random_state.choice(n_pop, n_select)
            reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
            while np.any(reselect):
                P[reselect, j] = random_state.choice(n_pop, reselect.sum())
                reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
        return P

    def _best(self, pop, n_select, n_parents, random_state):
        n_pop = len(pop)
        P = np.empty([n_select, n_parents], dtype=int)
        P[:, 0] = np.arange(n_pop)
        P[:, 1] = 0  # assumes population sorted best-first (FitnessSurvival)
        for j in range(2, n_parents):
            P[:, j] = random_state.choice(n_pop, n_select)
            reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
            while np.any(reselect):
                P[reselect, j] = random_state.choice(n_pop, reselect.sum())
                reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
        return P

    def _current_to_best(self, pop, n_select, n_parents, random_state):
        n_pop = len(pop)
        P = np.empty([n_select, n_parents], dtype=int)
        P[:, 0] = np.arange(n_pop)   # target
        P[:, 1] = np.arange(n_pop)   # current (mutation base)
        P[:, 2] = 0                   # best
        P[:, 3] = np.arange(n_pop)   # current (first of second diff pair)
        for j in range(4, n_parents):
            P[:, j] = random_state.choice(n_pop, n_select)
            reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
            while np.any(reselect):
                P[reselect, j] = random_state.choice(n_pop, reselect.sum())
                reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
        return P

    def _current_to_rand(self, pop, n_select, n_parents, random_state):
        n_pop = len(pop)
        P = np.empty([n_select, n_parents], dtype=int)
        P[:, 0] = np.arange(n_pop)   # target
        P[:, 1] = np.arange(n_pop)   # current (mutation base)
        P[:, 3] = np.arange(n_pop)   # current (first of second diff pair)
        P[:, 2] = random_state.choice(n_pop, n_select)
        reselect = (P[:, 2].reshape([-1, 1]) == P[:, [0, 1, 3]]).any(axis=1)
        while np.any(reselect):
            P[reselect, 2] = random_state.choice(n_pop, reselect.sum())
            reselect = (P[:, 2].reshape([-1, 1]) == P[:, [0, 1, 3]]).any(axis=1)
        for j in range(4, n_parents):
            P[:, j] = random_state.choice(n_pop, n_select)
            reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
            while np.any(reselect):
                P[reselect, j] = random_state.choice(n_pop, reselect.sum())
                reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
        return P


def _ranks_from_pop(pop):
    ranks = pop.get("rank")
    none_mask = np.array([r is None for r in ranks])
    if np.any(none_mask):
        ranks = ranks.copy()
        ranks[none_mask] = np.arange(len(pop))[none_mask]
    return ranks.astype(int)


def _rank_sort(P, pop):
    ranks = _ranks_from_pop(pop)
    sorted_idx = np.argsort(ranks[P], axis=1, kind="stable")
    S = np.take_along_axis(P, sorted_idx, axis=1)
    n_diffs = P.shape[1] // 2
    result = P.copy()
    result[:, 0] = S[:, 0]
    for j in range(1, n_diffs + 1):
        result[:, 2 * j - 1] = S[:, j]
        result[:, 2 * j] = S[:, -j]
    return result
