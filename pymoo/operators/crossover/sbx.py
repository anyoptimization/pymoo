import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.variable import Real, get
from pymoo.operators.repair.bounds_repair import repair_clamp
from pymoo.util import default_random_state


# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------


@default_random_state
def cross_sbx(X, xl, xu, eta, prob_var, prob_bin, eps=1.0e-14, random_state=None):
    n_parents, n_matings, n_var = X.shape

    # the probability of a crossover for each of the variables
    cross = random_state.random((n_matings, n_var)) < prob_var

    # when solutions are too close -> do not apply sbx crossover
    too_close = np.abs(X[0] - X[1]) <= eps

    # disable if two individuals are already too close
    cross[too_close] = False

    # disable crossover when lower and upper bound are identical
    cross[:, xl == xu] = False

    # preserve parent identity while getting values for SBX calculation
    p1 = X[0][cross]
    p2 = X[1][cross]
    
    # assign y1 the smaller and y2 the larger value for SBX calculation
    sm = p1 < p2
    y1 = np.where(sm, p1, p2)
    y2 = np.where(sm, p2, p1)

    # mask all the values that should be crossovered
    _xl = np.repeat(xl[None, :], n_matings, axis=0)[cross]
    _xu = np.repeat(xu[None, :], n_matings, axis=0)[cross]
    eta = eta.repeat(n_var, axis=1)[cross]
    prob_bin = prob_bin.repeat(n_var, axis=1)[cross]

    # random values for each individual
    rand = random_state.random(len(eta))

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

    # assign children based on parent position, then apply exchange probability
    child1 = np.where(sm, c1, c2)  # child for parent 1
    child2 = np.where(sm, c2, c1)  # child for parent 2
    
    # exchange children with given probability
    b = np.bitwise_xor(
        random_state.random(len(prob_bin)) < prob_bin,
        X[0, cross] > X[1, cross]
    )
    child1, child2 = np.where(b, (child2, child1), (child1, child2))

    # first copy the unmodified parents
    Q = np.copy(X)

    # copy the positions where the crossover was done
    Q[0, cross] = child1
    Q[1, cross] = child2

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
                 prob_exch=1.0,
                 prob_bin=0.5,
                 n_offsprings=2,
                 **kwargs):
        super().__init__(2, n_offsprings, **kwargs)

        self.prob_var = Real(prob_var, bounds=(0.1, 0.9))
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, None))
        self.prob_exch = Real(prob_exch, bounds=(0.0, 1.0), strict=(0.0, 1.0))
        self.prob_bin = Real(prob_bin, bounds=(0.0, 1.0), strict=(0.0, 1.0))

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        _, n_matings, _ = X.shape

        # get the parameters required by SBX
        eta, prob_var, prob_exch, prob_bin = get(self.eta, self.prob_var, self.prob_exch, self.prob_bin,
                                                 size=(n_matings, 1))

        # set the binomial probability to zero if no exchange between individuals shall happen
        rand = random_state.random((len(prob_bin), 1))
        prob_bin[rand > prob_exch] = 0.0

        Q = cross_sbx(X.astype(float), problem.xl, problem.xu, eta, prob_var, prob_bin, random_state=random_state)

        if self.n_offsprings == 1:
            rand = random_state.random(size=n_matings) < 0.5
            Q[0, rand] = Q[1, rand]
            Q = Q[[0]]

        return Q


class SBX(SimulatedBinaryCrossover):
    pass
