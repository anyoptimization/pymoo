import numpy as np

from pymoo.core.operator import Operator
from pymoo.core.population import Population
from pymoo.core.variable import Real, get


class Crossover(Operator):

    def __init__(self,
                 n_parents,
                 n_offsprings,
                 prob=0.9,
                 **kwargs):
        super().__init__()
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings
        self.prob = Real(prob, bounds=(0.7, 1.0), strict=(0.0, 1.0))

    def do(self, problem, pop, parents=None, **kwargs):
        n_parents, n_offsprings = self.n_parents, self.n_offsprings

        # get the actual design space values from the parents for the implementation
        if parents is None:
            X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1).copy()
            n_matings = len(pop)
        else:
            X = np.swapaxes(pop.get("X")[parents], 0, 1)
            n_matings = len(parents)

        # the array where the offsprings will be stored to
        Xp = np.zeros((n_offsprings, n_matings, problem.n_var), dtype=X.dtype)

        # the probability of executing the crossover
        prob = get(self.prob, size=n_matings)

        # a boolean mask when crossover is actually executed
        cross = np.random.random(n_matings) < prob

        # the design space from the parents used for the crossover
        if cross.sum() > 0:
            Q = self._do(problem, X, **kwargs)
            assert Q.shape == (n_offsprings, n_matings, problem.n_var), "Shape is incorrect of crossover impl."
            Xp[:, cross] = Q[:, cross]

        # now set the parents whenever no crossover has been applied
        for k in np.where(~cross)[0]:
            if n_offsprings < n_parents:
                s = np.random.choice(np.arange(self.n_parents), size=n_offsprings, replace=False)
            elif n_offsprings == n_parents:
                s = np.arange(n_parents)
            else:
                s = []
                while len(s) < n_offsprings:
                    s.extend(np.random.permutation(n_parents))
                s = s[:n_offsprings]

            Xp[:, k] = X[s, k]

        # flatten the array to become a 2d-array
        Xp = Xp.reshape(-1, X.shape[-1])

        # create a population object
        off = Population.new("X", Xp)

        return off

    def _do(self, problem, X, **kwargs):
        pass


def cross_norm_prob_var(n_var, max_perc=None):
    if max_perc is None:
        max_perc = 1 - 1 / n_var

    def norm_to_prob(x):
        prob_min = 1 / n_var
        prob_max = max(prob_min + 1e-16, max_perc)
        return prob_min + x * (prob_max - prob_min)

    return norm_to_prob


class VariableWiseCrossover(Crossover):

    def __init__(self, n_parents, n_offsprings, prob_var=0.5, **kwargs):
        super().__init__(n_parents, n_offsprings, **kwargs)
        self.prob_var = Real(prob_var, bounds=(0.00, 0.5))


class ElementwiseCrossover(Crossover):
    """
    The purpose of this crossover meta-crossover is the convenience of applying the operation an individual level
    (elementwise). This wrapper transform all crossovers in pymoo to such an operation.
    """

    def __init__(self, crossover, **kwargs):
        super().__init__(**kwargs)
        self.crossover = crossover

    def do(self, problem, *args, **kwargs):
        pop = Population.create(*args)
        parents = np.arange(len(args))[None, :]
        return self.crossover.do(problem, pop, parents, **kwargs)
