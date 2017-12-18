import numpy as np


class BinaryRandomSampling:
    def sample(self, p, n):
        return np.random.randint(2, size=(n, p.n_var))
