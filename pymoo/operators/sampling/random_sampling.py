from pymoo.model.sampling import Sampling

import numpy as np

from pymoo.util.normalization import denormalize


class FloatRandomSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return denormalize(val, problem.xl, problem.xu)


class BinaryRandomSampling(Sampling):

    def __init__(self) -> None:
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(np.bool)
