import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.algorithm import Algorithm
from pymoo.operators.crossover.real_differental_evolution_crossover import DifferentalEvolutionCrossover
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.rand import random
from pymoo.util.misc import repair
from pymoo.util.reference_directions import get_ref_dirs_from_n


class DifferentalEvolution(Algorithm):
    def __init__(self,
                 pop_size=100,
                 sampling=RealRandomSampling(),
                 F_CR=1,
                 F=0.85,
                 variant="DE/rand/1",
                 verbose=False,
                 callback=None):
        super().__init__()

        self.pop_size = pop_size
        self.sampling = sampling
        self.F_CR = F_CR
        self.F = F
        self.variant = variant

        self.best_i = None

        self.verbose = verbose
        self.callback = callback

    def _solve(self, problem, evaluator):

        # create the population according to the factoring strategy
        if isinstance(self.sampling, np.ndarray):
            X = self.sampling
        else:
            X = self.sampling.sample(problem, self.pop_size, self)

        # evaluate and find ideal point
        F, _ = evaluator.eval(problem, X)
        self.best_i = np.argmin(F)

        while evaluator.has_next():

            # random permutation for the crossover
            P = np.reshape(np.concatenate([random.perm(self.pop_size) for _ in range(3)]), (-1, 3))

            # do the crossover
            r = random.random((self.pop_size, problem.n_var)) < self.F_CR
            off_X = np.copy(X)

            if self.variant == "DE/rand/1":
                off_X[r] = (X[P[:, 2], :] + self.F * (X[P[:, 0], :] - X[P[:, 1], :]))[r]
                off_origin = P[:, 2]
            elif self.variant == "DE/local-to-best/1":
                off_X = X + self.F * (X[self.best_i, :] - X) + self.F * (X[P[:, 0], :] - X[P[:, 1], :])
                off_origin = np.arange(self.pop_size)
            elif self.variant == "DE/best/1":
                sc = self.F + random.random(problem.n_var) * 1e-4
                off_X = X[self.best_i, :] + (X[P[:, 0], :] - X[P[:, 1], :]) * sc
                off_origin = np.full(self.pop_size, self.best_i)
            elif self.variant == "DE/best/1 dfdf":
                pass
            else:
                raise Exception("DE variant %s not known." % self.variant)

            # repair if not in design space bounds anymore
            off_X = repair(off_X, problem.xl, problem.xu)

            # evaluate the offsprings
            off_F, _ = evaluator.eval(problem, off_X)

            # replace if better
            off_is_better = np.where(off_F < F)[0]
            F[off_is_better, :] = off_F[off_is_better, :]
            X[off_is_better, :] = off_X[off_is_better, :]

            # find the index of the best individual
            self.best_i = np.argmin(F)

            # self._do_each_generation()

        print(F[np.argmin(F), :])

        return X, F, np.zeros(self.pop_size)[:, None]
