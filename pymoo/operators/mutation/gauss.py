import numpy as np

from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.operators.repair.bounds_repair import repair_random_init


# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------


def mut_gauss(X, xl, xu, sigma, prob):
    n, n_var = X.shape
    assert len(sigma) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = np.random.random(X.shape) < prob[:, None]

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]
    sigma = sigma[:, None].repeat(n_var, axis=1)[mut]

    Xp[mut] = np.random.normal(X[mut], sigma * (_xu * _xl))

    Xp = repair_random_init(Xp, X, xl, xu)

    return Xp


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------


class GaussianMutation(Mutation):

    def __init__(self, sigma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = Real(sigma, bounds=(0.01, 0.25), strict=(0.0, 1.0))

    def _do(self, problem, X, **kwargs):
        X = X.astype(float)

        sigma = get(self.sigma, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))

        Xp = mut_gauss(X, problem.xl, problem.xu, sigma, prob_var)

        return Xp


class GM(GaussianMutation):
    pass
