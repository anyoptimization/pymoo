import numpy as np

from pymoo.util.misc import cdist


class DuplicateElimination:

    def __init__(self, func=lambda pop: pop.get("X")) -> None:
        super().__init__()
        self.func = func

    def do(self, pop, *args, return_indices=False, to_itself=True):
        original = pop

        if to_itself:
            pop = pop[~self._do(pop, None, np.full(len(pop), False))]

        for arg in args:
            if len(arg) > 0:

                if len(pop) == 0:
                    break
                elif len(arg) == 0:
                    continue
                else:
                    pop = pop[~self._do(pop, None, np.full(len(pop), False))]


        if return_indices:

            H = {}
            for k, ind in enumerate(original):
                H[ind] = k

            no_duplicate = [H[ind] for ind in pop]
            is_duplicate = [i for i in range(len(original)) if i not in no_duplicate]

            return pop, no_duplicate, is_duplicate
        else:
            return pop

    def _do(self, pop, other, is_duplicate):
        pass


class DefaultDuplicateElimination(DuplicateElimination):

    def __init__(self, epsilon=1e-16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def calc_dist(self, pop, other=None):
        X = self.func(pop)

        if other is None:
            D = cdist(X, X)
            D[np.triu_indices(len(X))] = np.inf
        else:
            _X = self.func(other)
            D = cdist(X, _X)

        return D

    def _do(self, pop, other, is_duplicate):
        D = self.calc_dist(pop, other)
        D[np.isnan(D)] = np.inf

        is_duplicate[np.any(D < self.epsilon, axis=1)] = True
        return is_duplicate


class ElementwiseDuplicateElimination(DefaultDuplicateElimination):

    def __init__(self, cmp=None, epsilon=1e-16, **kwargs) -> None:
        super().__init__(epsilon, **kwargs)
        self.cmp = cmp
        if self.cmp is None:
            self.cmp = lambda a, b: self.compare(a, b)

    def compare(self, a, b):
        pass

    def calc_dist(self, pop, other=None):
        if other is None:
            D = np.full((len(pop), len(pop)), np.inf)
            for i in range(len(pop)):
                for j in range(i + 1, len(pop)):
                    val = self.cmp(pop[i], pop[j])
                    if isinstance(val, bool):
                        val = 0.0 if val else 1.0
                    D[i, j] = val
            D = D.T

        else:
            D = np.full((len(pop), len(other)), np.inf)
            for i in range(len(pop)):
                for j in range(len(other)):
                    val = self.cmp(pop[i], other[j])
                    D[i, j] = val

        return D


def to_hash(x):
    try:
        h = hash(x)
    except:
        try:
            h = hash(str(x))
        except:
            raise Exception("Hash could not be calculated. Please use another duplicate elimination.")

    return h


class HashDuplicateElimination(DuplicateElimination):

    def __init__(self, func=to_hash) -> None:
        super().__init__()
        self.func = func

    def _do(self, pop, other, is_duplicate):
        H = set()

        if other is not None:
            for o in other:
                val = self.func(o)
                H.add(self.func(val))

        for i, ind in enumerate(pop):
            val = self.func(ind)
            h = self.func(val)

            if h in H:
                is_duplicate[i] = True
            else:
                H.add(h)

        return is_duplicate
