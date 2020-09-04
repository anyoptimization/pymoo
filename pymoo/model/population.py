import numpy as np

from pymoo.model.individual import Individual


def interleaving_args(*args, kwargs=None):
    if len(args) % 2 != 0:
        raise Exception(f"Even number of arguments are required but {len(args)} arguments were provided.")

    if kwargs is None:
        kwargs = {}

    for i in range(int(len(args) / 2)):
        key, values = args[i * 2], args[i * 2 + 1]
        kwargs[key] = values
    return kwargs


class Population(np.ndarray):

    def __new__(cls, n_individuals=0):
        obj = super(Population, cls).__new__(cls, n_individuals, dtype=cls).view(cls)
        for i in range(n_individuals):
            obj[i] = Individual()
        return obj

    def copy(self, deep=False):
        pop = Population(n_individuals=len(self))
        for i in range(len(self)):
            pop[i] = self[i].copy(deep=deep)
        return pop

    def has(self, key):
        return all([ind.has(key) for ind in self])

    def collect(self, func, to_numpy=True):
        val = []
        for i in range(len(self)):
            val.append(func(self[i]))
        if to_numpy:
            val = np.array(val)
        return val

    def set(self, *args, **kwargs):

        # if population is empty just return
        if self.size == 0:
            return

        # done for the old interface with the interleaving variable definition
        kwargs = interleaving_args(*args, kwargs=kwargs)

        # for each entry in the dictionary set it to each individual
        for key, values in kwargs.items():
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

        # for each individual
        for i in range(len(self)):

            # for each argument
            for c in args:
                val[c].append(self[i].get(c))

        # convert the results to a list
        res = [val[c] for c in args]

        # to numpy array if desired - default true
        if to_numpy:
            res = [np.array(e) for e in res]

        # return as tuple or single value
        if len(args) == 1:
            return res[0]
        else:
            return tuple(res)

    def __deepcopy__(self, memo):
        return self.copy(deep=True)

    @classmethod
    def merge(cls, a, b):
        a, b = pop_from_array_or_individual(a), pop_from_array_or_individual(b)

        if len(a) == 0:
            return b
        elif len(b) == 0:
            return a
        else:
            obj = np.concatenate([a, b]).view(Population)
            return obj

    @classmethod
    def create(cls, *args):
        pop = np.concatenate([pop_from_array_or_individual(arg) for arg in args]).view(Population)
        return pop

    @classmethod
    def new(cls, *args, **kwargs):

        if len(args) == 1:
            return Population(n_individuals=args[0], **kwargs)
        else:
            kwargs = interleaving_args(*args, kwargs=kwargs)
            iterable = [v for _, v in kwargs.items() if hasattr(v, '__len__') and not isinstance(v, str)]
            if len(iterable) == 0:
                return Population()
            else:
                n = np.unique(np.array([len(v) for v in iterable]))

                if len(n) == 1:
                    n = n[0]
                    pop = Population(n_individuals=n)
                    pop.set(*args, **kwargs)
                    return pop
                else:
                    raise Exception(f"Population.new needs to be called with same-sized inputs, but the sizes are {n}")


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
