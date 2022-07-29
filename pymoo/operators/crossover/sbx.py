import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.variable import Real, get
from pymoo.operators.repair.bounds_repair import repair_clamp


# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------


def cross_sbx(X, xl, xu, eta, prob_var, prob_bin, eps=1.0e-14):
    n_parents, n_matings, n_var = X.shape

    # the probability of a crossover for each of the variables
    cross = np.random.random((n_matings, n_var)) < prob_var

    # when solutions are too close -> do not apply sbx crossover
    too_close = np.abs(X[0] - X[1]) <= eps

    # disable if two individuals are already too close
    cross[too_close] = False

    # disable crossover when lower and upper bound are identical
    cross[:, xl == xu] = False

    # assign y1 the smaller and y2 the larger value
    y1 = np.min(X, axis=0)[cross]
    y2 = np.max(X, axis=0)[cross]

    # mask all the values that should be crossovered
    _xl = np.repeat(xl[None, :], n_matings, axis=0)[cross]
    _xu = np.repeat(xu[None, :], n_matings, axis=0)[cross]
    eta = eta.repeat(n_var, axis=1)[cross]
    prob_bin = prob_bin.repeat(n_var, axis=1)[cross]

    # random values for each individual
    rand = np.random.random(len(eta))

    def calc_betaq(beta):
        alpha = 2.0 - np.power(beta, -(eta + 1.0))

        mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))

        betaq = np.zeros(mask.shape)
        betaq[mask] = np.power((rand * alpha), (1.0 / (eta + 1.0)))[mask]
        betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (eta + 1.0)))[mask_not]

        return betaq

    # difference between all variables
    delta = (y2 - y1)

    beta = 1.0 + (2.0 * (y1 - _xl) / delta)
    betaq = calc_betaq(beta)
    c1 = 0.5 * ((y1 + y2) - betaq * delta)

    beta = 1.0 + (2.0 * (_xu - y2) / delta)
    betaq = calc_betaq(beta)
    c2 = 0.5 * ((y1 + y2) + betaq * delta)

    # with the given probability either assign the value from the first or second parent
    b = np.random.random(len(prob_bin)) < prob_bin
    tmp = np.copy(c1[b])
    c1[b] = c2[b]
    c2[b] = tmp

    # first copy the unmodified parents
    Q = np.copy(X)

    # copy the positions where the crossover was done
    Q[0, cross] = c1
    Q[1, cross] = c2

    Q[0] = repair_clamp(Q[0], xl, xu)
    Q[1] = repair_clamp(Q[1], xl, xu)

    return Q


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------


class SimulatedBinaryCrossover(Crossover):

    def __init__(self,
                 prob_var=0.5,
                 eta=15,
                 prob_exch=0.5,
                 prob_bin=0.5,
                 n_offsprings=2,
                 **kwargs):
        super().__init__(2, n_offsprings, **kwargs)

        self.prob_var = Real(prob_var, bounds=(0.1, 0.9))
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, None))
        self.prob_exch = Real(prob_exch, bounds=(0.0, 1.0), strict=(0.0, 1.0))
        self.prob_bin = Real(prob_bin, bounds=(0.0, 1.0), strict=(0.0, 1.0))

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape

        # get the parameters required by SBX
        eta, prob_var, prob_exch, prob_bin = get(self.eta, self.prob_var, self.prob_exch, self.prob_bin,
                                                 size=(n_matings, 1))

        # set the binomial probability to zero if no exchange between individuals shall happen
        rand = np.random.random((len(prob_bin), 1))
        prob_bin[rand > prob_exch] = 0.0

        Q = cross_sbx(X.astype(float), problem.xl, problem.xu, eta, prob_var, prob_bin)

        if self.n_offsprings == 1:
            rand = np.random.random(size=n_matings) < 0.5
            Q[0, rand] = Q[1, rand]
            Q = Q[[0]]

        return Q


class SBX(SimulatedBinaryCrossover):
    pass
