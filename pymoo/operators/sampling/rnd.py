import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util.normalization import denormalize


def random_by_bounds(n_var, xl, xu, n_samples=1):
    val = np.random.random((n_samples, n_var))
    return denormalize(val, xl, xu)


def random(problem, n_samples=1):
    return random_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples=n_samples)


class FloatRandomSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=np.float64) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        return random(problem, n_samples=n_samples)


class BinaryRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)


class PermutationRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), 0, dtype=int)
        for i in range(n_samples):
            X[i, :] = np.random.permutation(problem.n_var)
        return X
