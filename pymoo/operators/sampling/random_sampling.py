import numpy as np

from pymoo.model.sampling import Sampling
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

    def __init__(self, var_type=np.float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        return random(problem, n_samples=n_samples)


class BinaryRandomSampling(Sampling):

    def __init__(self) -> None:
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(np.bool)
