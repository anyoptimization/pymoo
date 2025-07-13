from math import sqrt, log, exp

import numpy as np

from pymoo.algorithms.soo.nonconvex.es import es_sigma, es_mut_repair
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string


class ISRES(SRES):

    def __init__(self, gamma=0.85, alpha=0.2, **kwargs):
        """
        Improved Stochastic Ranking Evolutionary Strategy (SRES)

        Parameters
        ----------
        alpha : float
            Length scale of the differentials during mutation.
        PF: float
            The stochastic ranking weight for choosing a random decision while doing the modified bubble sort.
        """

        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        n = problem.n_var

        chi = (1 / (2 * n) + 1 / (2 * (n ** 0.5)))
        varphi = sqrt((2 / chi) * log((1 / self.alpha) * (exp(self.phi ** 2 * chi / 2) - (1 - self.alpha))))

        self.taup = varphi / ((2 * n) ** 0.5)
        self.tau = varphi / ((2 * (n ** 0.5)) ** 0.5)

    def _infill(self):
        pop, mu, _lambda = self.pop, self.pop_size, self.n_offsprings
        xl, xu = self.problem.bounds()
        X, sigma = pop.get("X", "sigma")

        # cycle through the elites individuals for create the solutions
        I = np.arange(_lambda) % min(mu, len(X))

        # transform X and sigma to the shape of number of offsprings
        X, sigma = X[I], sigma[I]

        # copy the original sigma to sigma prime to be modified
        Xp, sigmap = np.copy(X), np.copy(sigma)

        # for the best individuals do differential variation to provide a direction to search in
        Xp[:mu - 1] = X[:mu - 1] + self.gamma * (X[0] - X[1:mu])

        # update the sigma values for elite and non-elite individuals
        sigmap[mu - 1:] = np.minimum(self.sigma_max, es_sigma(sigma[mu - 1:], self.tau, self.taup, random_state=self.random_state))

        # execute the evolutionary strategy to calculate the offspring solutions
        Xp[mu - 1:] = X[mu - 1:] + sigmap[mu - 1:] * self.random_state.normal(size=sigmap[mu - 1:].shape)

        # repair the individuals which are not feasible by sampling from sigma again
        Xp = es_mut_repair(Xp, X, sigmap, xl, xu, 10, random_state=self.random_state)

        # now update the sigma values of the non-elites only
        sigmap[mu:] = sigma[mu:] + self.alpha * (sigmap[mu:] - sigma[mu:])

        # create the population to proceed further
        off = Population.new(X=Xp, sigma=sigmap)

        return off


parse_doc_string(ISRES.__init__)
