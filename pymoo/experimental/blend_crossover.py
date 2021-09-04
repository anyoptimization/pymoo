import numpy as np

from pymoo.core.crossover import Crossover


class BlendCrossover(Crossover):
    def __init__(self, alpha, prob_per_variable=0.5, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.alpha = float(alpha)
        self.prob_per_variable = prob_per_variable

    def _do(self, problem, X, **kwargs):

        X = X.astype(float)
        _, n_matings, n_var = X.shape

        # boundaries of the problem
        xl, xu = problem.xl, problem.xu

        # crossover mask that will be used in the end
        do_crossover = np.full(X[0].shape, True)

        # per variable the probability is then 50%
        do_crossover[np.random.random((n_matings, problem.n_var)) > self.prob_per_variable] = False
        # also if values are too close no mating is done
        do_crossover[np.abs(X[0] - X[1]) <= 1.0e-14] = False

        d = np.abs(X[0, :, :] - X[1, :, :])
        y1 = np.min(X, axis=0)
        y2 = np.max(X, axis=0)

        _min = y1 - self.alpha * d
        _max = y2 + self.alpha * d

        c = np.random.uniform(_min, _max, X.shape)
        # for i in range(X.shape[1]):
        #     for j in range(X.shape[2]):
        #         c[:, i, j] = np.random.uniform(y1[i, j] - self.alpha*d[i, j], y2[i, j] + self.alpha*d[i, j], 2)

        return c
