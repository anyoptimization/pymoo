import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, eta, n_offsprings=2, prob_per_variable=0.5, **kwargs):
        super().__init__(2, n_offsprings, **kwargs)
        self.eta = float(eta)
        self.prob_per_variable = prob_per_variable

    def _do(self, problem, X, **kwargs):

        X = X.astype(float)
        _, n_matings, n_var = X.shape

        # boundaries of the problem
        xl, xu = problem.xl, problem.xu

        #if np.any(X < xl) or np.any(X > xu):
        #    raise Exception("Simulated binary crossover requires all variables to be in bounds!")

        # crossover mask that will be used in the end
        do_crossover = np.full(X[0].shape, True)

        # per variable the probability is then 50%
        do_crossover[np.random.random((n_matings, problem.n_var)) > self.prob_per_variable] = False
        # also if values are too close no mating is done
        do_crossover[np.abs(X[0] - X[1]) <= 1.0e-14] = False

        # assign y1 the smaller and y2 the larger value
        y1 = np.min(X, axis=0)
        y2 = np.max(X, axis=0)

        # random values for each individual
        rand = np.random.random((n_matings, problem.n_var))

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
        # delta[np.logical_or(delta < 1.0e-10, np.logical_not(do_crossover))] = 1.0e-10
        delta[delta < 1.0e-10] = 1.0e-10

        beta = 1.0 + (2.0 * (y1 - xl) / delta)
        betaq = calc_betaq(beta)
        c1 = 0.5 * ((y1 + y2) - betaq * delta)

        beta = 1.0 + (2.0 * (xu - y2) / delta)
        betaq = calc_betaq(beta)
        c2 = 0.5 * ((y1 + y2) + betaq * delta)

        # do randomly a swap of variables
        b = np.random.random((n_matings, problem.n_var)) <= 0.5
        val = np.copy(c1[b])
        c1[b] = c2[b]
        c2[b] = val

        # take the parents as _template
        c = np.copy(X)

        # copy the positions where the crossover was done
        c[0, do_crossover] = c1[do_crossover]
        c[1, do_crossover] = c2[do_crossover]

        c[0] = set_to_bounds_if_outside_by_problem(problem, c[0])
        c[1] = set_to_bounds_if_outside_by_problem(problem, c[1])

        if self.n_offsprings == 1:
            # Randomly select one offspring
            c = c[np.random.choice(2, X.shape[1]), np.arange(X.shape[1])]
            c = c.reshape((1, X.shape[1], X.shape[2]))

        return c


class SBX(SimulatedBinaryCrossover):
    pass
