import numpy as np

from pymoo.model.individual import Individual


class Population(np.ndarray):

    def __new__(cls, n_individuals=0, individual=Individual()):
        obj = super(Population, cls).__new__(cls, n_individuals, dtype=individual.__class__).view(cls)
        for i in range(n_individuals):
            obj[i] = individual.copy()
        obj.individual = individual
        return obj

    def merge(self, other):
        obj = np.concatenate([self, other]).view(Population)
        obj.individual = self.individual
        return obj

    def copy(self):
        pop = Population(n_individuals=len(self), individual=self.individual)
        for i in range(len(self)):
            pop[i] = self[i]
        return pop

    def __deepcopy__(self, memo):
        return self.copy()

    @classmethod
    def create(cls, *args):
        pop = np.concatenate([pop_from_array_or_individual(arg) for arg in args]).view(Population)
        pop.individual = Individual()
        return pop

    def new(self, *args):

        if len(args) == 1:
            return Population(n_individuals=args[0], individual=self.individual)
        else:
            n = len(args[1]) if len(args) > 0 else 0
            pop = Population(n_individuals=n, individual=self.individual)
            if len(args) > 0:
                pop.set(*args)
            return pop

    def collect(self, func, as_numpy_array=True):
        val = []
        for i in range(len(self)):
            val.append(func(self[i]))
        if as_numpy_array:
            val = np.array(val)
        return val

    def set(self, *args):

        for i in range(int(len(args) / 2)):

            key, values = args[i * 2], args[i * 2 + 1]

            if len(values) != len(self):
                raise Exception("Population Set Attribute Error: Number of values and population size do not match!")

            for i in range(len(values)):

                if key in self[i].__dict__:
                    self[i].__dict__[key] = values[i]
                else:
                    self[i].data[key] = values[i]

        return self

    def get(self, *args):

        val = {}
        for c in args:
            val[c] = []

        for i in range(len(self)):

            for c in args:

                if c in self[i].__dict__:
                    val[c].append(self[i].__dict__[c])
                elif c in self[i].data:
                    val[c].append(self[i].data[c])

        res = [np.array(val[c]) for c in args]

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
        pop = pop.new("X", [array])
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
