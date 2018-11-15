import numpy as np

from pymoo.model.mutation import Mutation
from pymoo.rand import random
from pymoo.util.misc import covert_to_type


class PolynomialMutation(Mutation):
    def __init__(self, eta_mut, prob_mut=None):
        super().__init__()
        self.eta_mut = float(eta_mut)
        if prob_mut is not None:
            self.prob_mut = float(prob_mut)
        else:
            self.prob_mut = None

    def _do(self, problem, pop, **kwargs):

        X = pop.get("X")
        Y = np.full(X.shape, np.inf)

        if self.prob_mut is None:
            self.prob_mut = 1.0 / problem.n_var

        do_mutation = random.random(X.shape) < self.prob_mut

        Y[:, :] = X

        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta_mut + 1.0)

        rand = random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta_mut + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta_mut + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        Y[do_mutation] = _Y

        return pop.new("X", Y)

