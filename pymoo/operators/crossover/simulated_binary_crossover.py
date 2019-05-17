import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.repair.out_of_bounds_repair import OutOfBoundsRepair
from pymoo.rand import random


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, prob, eta, prob_per_variable=0.5, var_type=np.double):
        super().__init__(2, 2)
        self.prob = float(prob)
        self.eta = float(eta)
        self.prob_per_variable = prob_per_variable
        self.var_type = var_type

    def _do(self, problem, pop, parents, **kwargs):

        n_matings = parents.shape[0]
        children = np.full((n_matings * self.n_offsprings, problem.n_var), np.inf)
        X = pop.get("X")[parents.T].astype(np.double)
        xl, xu = problem.xl, problem.xu

        # in case of an integer problem we do the same, but change the bounds slightly and
        # round later on
        if self.var_type == np.int:
            xl = problem.xl - 0.5
            xu = problem.xu + (0.5 - 1e-16)

        # crossover mask that will be used in the end
        do_crossover = np.full(X[0].shape, True)

        # simulating probability of doing a crossover with the parents at all
        do_crossover[random.random(n_matings) > self.prob, :] = False
        # per variable the probability is then 50%
        do_crossover[random.random((n_matings, problem.n_var)) > self.prob_per_variable] = False
        # also if values are too close no mating is done
        do_crossover[np.abs(X[0] - X[1]) <= 1.0e-14] = False

        # assign y1 the smaller and y2 the larger value
        y1 = np.min(X, axis=0)
        y2 = np.max(X, axis=0)

        # random values for each individual
        rand = random.random((n_matings, problem.n_var))

        def calc_betaq(beta):
            alpha = 2.0 - np.power(beta, -(self.eta + 1.0))

            mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))

            betaq = np.zeros(mask.shape)
            betaq[mask] = np.power((rand * alpha), (1.0 / (self.eta + 1.0)))[mask]
            betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (self.eta + 1.0)))[mask_not]

            return betaq

        # difference between all variables
        delta = (y2 - y1)

        # now just be sure not dividing by zero (these cases will be filtered later anyway)
        #delta[np.logical_or(delta < 1.0e-10, np.logical_not(do_crossover))] = 1.0e-10
        delta[delta < 1.0e-10] = 1.0e-10

        beta = 1.0 + (2.0 * (y1 - xl) / delta)
        betaq = calc_betaq(beta)
        c1 = 0.5 * ((y1 + y2) - betaq * delta)

        beta = 1.0 + (2.0 * (xu - y2) / delta)
        betaq = calc_betaq(beta)
        c2 = 0.5 * ((y1 + y2) + betaq * delta)

        # do randomly a swap of variables
        b = random.random((n_matings, problem.n_var)) <= 0.5
        val = c1[b]
        c1[b] = c2[b]
        c2[b] = val

        # take the parents as _template
        c = X.astype(np.double)

        # copy the positions where the crossover was done
        c[0, do_crossover] = c1[do_crossover]
        c[1, do_crossover] = c2[do_crossover]

        # copy to the structure which is returned
        children[:n_matings, :] = c[0]
        children[n_matings:, :] = c[1]

        if self.var_type == np.int:
            children = np.rint(children).astype(np.int)

        # now repair all the offsprings to be in bound
        off = OutOfBoundsRepair().do(problem, pop.new("X", children))

        return off
