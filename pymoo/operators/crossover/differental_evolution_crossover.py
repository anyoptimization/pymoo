import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.repair.inverse_penalty import InversePenaltyOutOfBoundsRepair


class DifferentialEvolutionCrossover(Crossover):

    def __init__(self,
                 weight=0.8,
                 dither=None,
                 jitter=False,
                 repair=InversePenaltyOutOfBoundsRepair(),
                 n_diffs=1,
                 **kwargs):
        super().__init__(1 + 2 * n_diffs, 1, **kwargs)
        self.n_diffs = n_diffs
        self.repair = repair
        self.weight = weight
        self.dither = dither
        self.jitter = jitter

    def _do(self, problem, X, **kwargs):

        n_parents, n_matings, n_var = X.shape

        # the actual equation generation to multiple differences to be added -> get the first parent of each mating
        _X = X[0]

        # for each difference
        for k in range(self.n_diffs):

            # get the default value for the weight vector F
            F = self.weight

            if self.dither == "vector":
                F = (self.weight + np.random.random(n_matings) * (1 - self.weight))[:, None]
            elif self.dither == "scalar":
                F = self.weight + np.random.random() * (1 - self.weight)

            # http://www.cs.ndsu.nodak.edu/~siludwig/Publish/papers/SSCI20141.pdf
            if self.jitter:
                gamma = 0.0001
                F = (self.weight * (1 + gamma * (np.random.random(n_matings) - 0.5)))[:, None]

            # get the start index for each difference to be added
            i = 1 + 2 * k

            # an add the difference to the first vector
            _X = _X + F * (X[i] - X[i + 1])

        _X = self.repair.do(problem, _X, P=X[0])

        return _X[None, ...]
