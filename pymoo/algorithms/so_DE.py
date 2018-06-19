import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.operators.crossover.real_differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.rand import random
from pymoo.util.misc import parameter_less_constraints


class DifferentialEvolution(Algorithm):
    def __init__(self,
                 pop_size=100,
                 sampling=RealRandomSampling(),
                 crossover=DifferentialEvolutionCrossover(prob=0.5, weight=0.75, variant="DE/best/1",
                                                          method="binomial"),
                 **kwargs):

        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.sampling = sampling
        self.crossover = crossover
        self.best_i = None

    def _solve(self, problem, evaluator):

        # create the population according to the factoring strategy
        if isinstance(self.sampling, np.ndarray):
            X = self.sampling
        else:
            X = self.sampling.sample(problem, self.pop_size, self)

        # evaluate and find ideal point
        F, CV = evaluator.eval(problem, X)
        F = parameter_less_constraints(F, CV)
        self.best_i = np.argmin(F)

        while evaluator.has_next():

            P = np.reshape(np.concatenate([random.perm(self.pop_size) for _ in range(self.crossover.n_parents)]),
                           (-1, self.crossover.n_parents))
            off_X = self.crossover.do(problem, X[P, :], X=X, best_i=self.best_i)

            # evaluate the offsprings
            off_F, off_CV = evaluator.eval(problem, off_X, return_constraints=2)
            off_F = parameter_less_constraints(off_F, off_CV)

            # replace if better
            off_is_better = np.where(off_F < F)[0]
            F[off_is_better, :] = off_F[off_is_better, :]
            X[off_is_better, :] = off_X[off_is_better, :]

            # find the index of the best individual
            self.best_i = np.argmin(F)

            if self.verbose:
                print(evaluator.counter, F[self.best_i, :])

        return X, F, np.zeros(self.pop_size)[:, None]
