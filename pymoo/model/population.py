import copy

import numpy as np

from pymoo.model.individual import Individual


class Population(np.ndarray):

    individual = None

    def __new__(cls, n_individuals=0, individual=None):

        # the default individual if not specific otherwise
        if individual is None:
            individual = Individual()

        obj = super(Population, cls).__new__(cls, n_individuals, dtype=individual.__class__).view(cls)
        for i in range(n_individuals):
            obj[i] = individual.copy()
        obj.individual = individual
        return obj

    @classmethod
    def merge(cls, a, b):
        a, b = pop_from_array_or_individual(a), pop_from_array_or_individual(b)

        if len(a) == 0:
            return b
        elif len(b) == 0:
            return a
        else:
            obj = np.concatenate([a, b]).view(Population)
            obj.individual = a.individual
            return obj

    def copy(self, deep=False):
        pop = Population(n_individuals=len(self), individual=self.individual)
        for i in range(len(self)):
            val = copy.deepcopy(self[i]) if deep else self[i]
            pop[i] = val
        return pop

    def has(self, key):
        return all([ind.has(key) for ind in self])

    def __deepcopy__(self, memo):
        return self.copy()

    @classmethod
    def create(cls, *args):
        pop = np.concatenate([pop_from_array_or_individual(arg) for arg in args]).view(Population)
        pop.individual = Individual()
        return pop

    @classmethod
    def new(cls, *args, **kwargs):

        if len(args) == 1:
            return Population(n_individuals=args[0], **kwargs)
        else:
            n = len(args[1]) if len(args) > 0 else 0
            pop = Population(n_individuals=n, **kwargs)
            if len(args) > 0:
                pop.set(*args)
            return pop

    def collect(self, func, to_numpy=True):
        val = []
        for i in range(len(self)):
            val.append(func(self[i]))
        if to_numpy:
            val = np.array(val)
        return val

    def set(self, *args):

        for i in range(int(len(args) / 2)):

            key, values = args[i * 2], args[i * 2 + 1]
            is_iterable = hasattr(values, '__len__') and not isinstance(values, str)

            if is_iterable and len(values) != len(self):
                raise Exception("Population Set Attribute Error: Number of values and population size do not match!")

            for i in range(len(self)):
                val = values[i] if is_iterable else values
                self[i].set(key, val)

        return self

    def get(self, *args, to_numpy=True):

        val = {}
        for c in args:
            val[c] = []

        for i in range(len(self)):

            for c in args:
                val[c].append(self[i].get(c))

        res = [val[c] for c in args]
        if to_numpy:
            res = [np.array(e) for e in res]

        if len(args) == 1:
            return res[0]
        else:
            return tuple(res)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.individual = getattr(obj, 'individual', None)


def pop_from_array_or_individual(array, pop=None):
    # the population type can be different - (different type of individuals)
    if pop is None:
        pop = Population()

    # provide a whole population object - (individuals might be already evaluated)
    if isinstance(array, Population):
        pop = array
    elif isinstance(array, np.ndarray):
        pop = pop.new("X", np.atleast_2d(array))
    elif isinstance(array, Individual):
        pop = Population(1)
        pop[0] = array
    else:
        return None

    return pop


if __name__ == '__main__':
    pop = Population(10)
    pop.get("F")
    pop.new()
    print("")
