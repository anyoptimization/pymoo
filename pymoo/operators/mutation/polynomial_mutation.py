import numpy as np

from pymoo.model.mutation import Mutation
from pymoo.operators.repair.out_of_bounds_repair import OutOfBoundsRepair
from pymoo.rand import random


class PolynomialMutation(Mutation):
    def __init__(self, eta, prob=None, var_type=np.double):
        super().__init__()
        self.eta = float(eta)
        self.var_type = var_type
        if prob is not None:
            self.prob = float(prob)
        else:
            self.prob = None

    def _do(self, problem, pop, **kwargs):

        X = pop.get("X").astype(np.double)
        Y = np.full(X.shape, np.inf)

        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        do_mutation = random.random(X.shape) < self.prob

        Y[:, :] = X

        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]

        if self.var_type == np.int:
            xl -= 0.5
            xu += (0.5 - 1e-16)

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        rand = random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        Y[do_mutation] = _Y

        if self.var_type == np.int:
            Y = np.rint(Y).astype(np.int)

        off = OutOfBoundsRepair().do(problem, pop.new("X", Y))

        return off
