import numpy as np

from pymoo.util.misc import cdist


def default_attr(pop):
    return pop.get("X")


class DuplicateElimination:

    def __init__(self, func=None) -> None:
        super().__init__()
        self.func = func

        if self.func is None:
            self.func = default_attr

    def do(self, pop, *args, return_indices=False, to_itself=True):
        original = pop

        if len(pop) == 0:
            return (pop, [], []) if return_indices else pop

        if to_itself:
            pop = pop[~self._do(pop, None, np.full(len(pop), False))]

        for arg in args:
            if len(arg) > 0:

                if len(pop) == 0:
                    break
                elif len(arg) == 0:
                    continue
                else:
                    pop = pop[~self._do(pop, arg, np.full(len(pop), False))]

        if return_indices:
            no_duplicate, is_duplicate = [], []
            H = set(pop)

            for i, ind in enumerate(original):
                if ind in H:
                    no_duplicate.append(i)
                else:
                    is_duplicate.append(i)

            return pop, no_duplicate, is_duplicate
        else:
            return pop

    def _do(self, pop, other, is_duplicate):
        return is_duplicate


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

        is_duplicate[np.any(D <= self.epsilon, axis=1)] = True
        return is_duplicate


def to_float(val):
    if isinstance(val, bool) or isinstance(val, np.bool_):
        return 0.0 if val else 1.0
    else:
        return val


class ElementwiseDuplicateElimination(DefaultDuplicateElimination):

    def __init__(self, cmp_func=None, **kwargs) -> None:
        super().__init__(**kwargs)

        if cmp_func is None:
            cmp_func = self.is_equal

        self.cmp_func = cmp_func

    def is_equal(self, a, b):
        pass

    def _do(self, pop, other, is_duplicate):

        if other is None:
            for i in range(len(pop)):
                for j in range(i + 1, len(pop)):
                    val = to_float(self.cmp_func(pop[i], pop[j]))
                    if val < self.epsilon:
                        is_duplicate[i] = True
                        break
        else:
            for i in range(len(pop)):
                for j in range(len(other)):
                    val = to_float(self.cmp_func(pop[i], other[j]))
                    if val < self.epsilon:
                        is_duplicate[i] = True
                        break

        return is_duplicate


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


class NoDuplicateElimination(DuplicateElimination):

    def do(self, pop, *args, **kwargs):
        return pop
