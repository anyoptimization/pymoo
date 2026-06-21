import numpy as np
from moocore import hv_approx, hv_contributions


class ApproximateHypervolume:

    def __init__(self, ref_point, n_samples=10000, method='Rphi-FWE+', seed=None) -> None:
        self.ref_point = ref_point
        self.n_samples = n_samples
        self.method = method
        self.seed = seed
        self.F = np.zeros((0, len(ref_point)))
        self.hv = 0.0
        self.hvc = np.zeros(0)

    def add(self, F):
        self.F = np.vstack([self.F, F])
        self._update()
        return self

    def delete(self, k):
        self.F = np.delete(self.F, k, axis=0)
        self._update()
        return self

    def _update(self):
        if len(self.F) == 0:
            self.hv = 0.0
            self.hvc = np.zeros(0)
        else:
            self.hv = hv_approx(self.F, ref=self.ref_point, nsamples=self.n_samples, method=self.method, seed=self.seed)
            self.hvc = hv_contributions(self.F, ref=self.ref_point)
