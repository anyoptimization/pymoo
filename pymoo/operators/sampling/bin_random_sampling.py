import numpy as np

from pymoo.rand import random


class BinaryRandomSampling:
    """
    Randomly sample a binary representation of 0's and 1's.
    """

    def sample(self, problem, pop, n_samples, **kwargs):
        val = random.randint(0, 2, size=(n_samples, problem.n_var)).astype(np.bool)
        return pop.new("X", val)
