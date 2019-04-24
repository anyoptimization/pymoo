from pymoo.model.sampling import Sampling
from pymoo.rand import random
import numpy as np

from pymoo.util.normalization import denormalize


class RandomSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=np.float) -> None:
        super().__init__()
        self.var_type = var_type

    def sample(self, problem, pop, n_samples, **kwargs):

        m = problem.n_var
        val = random.random(size=(n_samples, m))

        if self.var_type == np.bool:
            val = random.random((n_samples, m))
            val = (val < 0.5).astype(np.bool)
        elif self.var_type == np.int:
            val = np.rint(denormalize(val, problem.xl - 0.5, problem.xu + (0.5 - 1e-16))).astype(np.int)
        else:
            val = denormalize(val, problem.xl, problem.xu)

        return pop.new("X", val)

