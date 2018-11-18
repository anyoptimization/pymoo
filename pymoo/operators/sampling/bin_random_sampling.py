import numpy as np

from pymoo.model.sampling import Sampling
from pymoo.rand import random


class BinaryRandomSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def sample(self, problem, pop, n_samples, **kwargs):

        m = problem.n_var

        val = random.random((n_samples, m))
        val = (val < 0.5).astype(np.bool)

        return pop.new("X", val)
