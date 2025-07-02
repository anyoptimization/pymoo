import numpy as np

from pymoo.core.operator import Operator
from pymoo.core.population import Population
from pymoo.core.variable import Real, get
from pymoo.util import default_random_state


class Crossover(Operator):

    def __init__(self,
                 n_parents,
                 n_offsprings,
                 prob=0.9,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings
        self.prob = Real(prob, bounds=(0.5, 1.0), strict=(0.0, 1.0))

    @default_random_state
    def do(self, problem, pop, parents=None, *args, random_state=None, **kwargs):

        # if a parents with array with mating indices is provided -> transform the input first
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        # get the dimensions necessary to create in and output
        n_parents, n_offsprings = self.n_parents, self.n_offsprings
        n_matings, n_var = len(pop), problem.n_var

        # get the actual values from each of the parents
        X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1)
        if self.vtype is not None:
            X = X.astype(self.vtype)

        # the array where the offsprings will be stored to
        Xp = np.empty(shape=(n_offsprings, n_matings, n_var), dtype=X.dtype)

        # the probability of executing the crossover
        prob = get(self.prob, size=n_matings)

        # a boolean mask when crossover is actually executed
        cross = random_state.random(n_matings) < prob

        # the design space from the parents used for the crossover
        if np.any(cross):

            # we can not prefilter for cross first, because there might be other variables using the same shape as X
            Q = self._do(problem, X, *args, random_state=random_state, **kwargs)
            assert Q.shape == (n_offsprings, n_matings, problem.n_var), "Shape is incorrect of crossover impl."
            Xp[:, cross] = Q[:, cross]

        # now set the parents whenever NO crossover has been applied
        for k in np.flatnonzero(~cross):
            if n_offsprings < n_parents:
                s = random_state.choice(np.arange(self.n_parents), size=n_offsprings, replace=False)
            elif n_offsprings == n_parents:
                s = np.arange(n_parents)
            else:
                s = []
                while len(s) < n_offsprings:
                    s.extend(random_state.permutation(n_parents))
                s = s[:n_offsprings]

            Xp[:, k] = np.copy(X[s, k])

        # flatten the array to become a 2d-array
        Xp = Xp.reshape(-1, X.shape[-1])

        # create a population object
        off = Population.new("X", Xp)

        return off

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        pass


