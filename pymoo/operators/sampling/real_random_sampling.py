import numpy as np


class RealRandomSampling:
    def sample(self, p, n):
        m = len(p.xl)
        val = np.random.rand(n, m)
        for i in range(m):
            val[:, i] = val[:, i] * (p.xu[i] - p.xl[i]) + p.xl[i]
        return val
