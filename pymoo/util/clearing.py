import numpy as np


def func_select_by_objective(pop):
    F = pop.get("F")
    return F[:, 0].argmin()


def func_select_from_sorted(_):
    return 0


def select_by_clearing(pop, D, n_select, func_select, delta=0.05):
    clearing = EpsilonClearing(D, delta)

    while len(clearing.selected()) < n_select:
        remaining = clearing.remaining()

        if len(remaining) == 0:
            clearing.reset()
            remaining = clearing.remaining()

        best = remaining[func_select(pop[remaining])]
        clearing.select(best)

    S = clearing.selected()
    return S


class EpsilonClearing:

    def __init__(self,
                 D,
                 epsilon) -> None:

        super().__init__()

        if isinstance(D, tuple):
            self.n, self.D = D
        else:
            self.D = D
            self.n = len(D)

        self.epsilon = epsilon

        self.S = []
        self.C = np.full(self.n, False)

    def remaining(self):
        return np.where(~self.C)[0]

    def has_remaining(self):
        return self.C.sum() != self.n

    def cleared(self):
        return self.C

    def selected(self):
        return self.S

    def reset(self):
        self.C = np.full(self.n, False)
        self.C[self.S] = True

    def select(self, k):
        self.S.append(k)
        self.C[k] = True

        if callable(self.D):
            dist_to_other = self.D(k)
        else:
            dist_to_other = self.D[k]

        less_than_epsilon = dist_to_other < self.epsilon

        # problems which are currently not cleared and are supposed to
        cleared = np.where(np.logical_and(~self.C, less_than_epsilon))[0]

        # set them to be cleared
        self.C[cleared] = True

        return cleared
