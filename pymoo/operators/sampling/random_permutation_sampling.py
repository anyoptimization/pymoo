from pymoo.model.sampling import Sampling
import numpy as np


class PermutationRandomSampling(Sampling):
    """
    Generate random permutation.
    """
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), 0, dtype=np.int)
        for i in range(n_samples):
            X[i, :] = np.random.permutation(problem.n_var)
        return X