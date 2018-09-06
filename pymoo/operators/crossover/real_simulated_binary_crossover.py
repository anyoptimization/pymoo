import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.rand import random


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, prob_cross, eta_cross):
        super().__init__(2, 2)
        self.prob_cross = float(prob_cross)
        self.eta_cross = float(eta_cross)

    def _do(self, p, parents, children, D, **kwargs):
        n_matings = parents.shape[1]

        # crossover mask that will be used in the end
        do_crossover = np.full(parents[0].shape, True)

        # simulating probability of doing a crossover with the parents at all
        do_crossover[random.random(n_matings) > self.prob_cross, :] = False
        # per variable the probability is then 50%
        do_crossover[random.random((n_matings, p.n_var)) <= 0.5] = False
        # also if values are too close no mating is done
        do_crossover[np.abs(parents[0] - parents[1]) <= 1.0e-14] = False

        # assign y1 the smaller and y2 the larger value
        y1 = np.min(parents, axis=0)
        y2 = np.max(parents, axis=0)

        # random values for each individual
        rand = random.random((n_matings, p.n_var))

        def calc_betaq(beta):
            alpha = 2.0 - np.power(beta, -(self.eta_cross + 1.0))

            mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))

            betaq = np.zeros(mask.shape)
            betaq[mask] = np.power((rand * alpha), (1.0 / (self.eta_cross + 1.0)))[mask]
            betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (self.eta_cross + 1.0)))[mask_not]

            return betaq

        # difference between all variables
        delta = (y2 - y1)

        # now just be sure not dividing by zero (these cases will be filtered later anyway)
        delta[delta < 1.0e-14] = 1.0e-14

        beta = 1.0 + (2.0 * (y1 - p.xl) / delta)
        betaq = calc_betaq(beta)
        c1 = 0.5 * ((y1 + y2) - betaq * delta)

        beta = 1.0 + (2.0 * (p.xu - y2) / delta)
        betaq = calc_betaq(beta)
        c2 = 0.5 * ((y1 + y2) + betaq * delta)

        # do randomly a swap of variables
        b = random.random((n_matings, p.n_var)) <= 0.5
        val = c1[b]
        c1[b] = c2[b]
        c2[b] = val

        # take the parents as template
        c = parents

        # copy the positions where the crossover was done
        c[0, do_crossover] = c1[do_crossover]
        c[1, do_crossover] = c2[do_crossover]

        # copy to the structure which is returned
        children[:n_matings, :] = c[0]
        children[n_matings:, :] = c[1]

        # just be sure we are not out of bounds
        children[children < p.xl] = np.repeat(p.xl[None, :], children.shape[0], axis=0)[children < p.xl]
        children[children > p.xu] = np.repeat(p.xu[None, :], children.shape[0], axis=0)[children > p.xu]
