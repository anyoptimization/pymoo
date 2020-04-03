import abc

import numpy as np


class Indicator:

    def __init__(self, pf=None, ref_point=None, normalize=False, bounds=None):
        self.default_if_empty = 0.0
        self.pf = pf
        self.normalize = normalize
        self.bounds = bounds

        self.n_dim = None
        self.range = None

        if ref_point is None and pf is not None:
            self.ref_point = pf.max(axis=0)
        else:
            self.ref_point = ref_point

    def calc(self, F):

        # if it is a 1d array
        if F.ndim == 1:
            F = F[None, :]

        # if there are not data
        if F.shape[1] == 0:
            return self.default_if_empty

        _, self.n_dim = F.shape

        if self.normalize:
            if self.pf is not None:
                self.ideal_point = np.min(self.pf, axis=0)
                self.nadir_point = np.max(self.pf, axis=0)
            elif self.bounds is not None:
                self.ideal_point = self.bounds[0]
                self.nadir_point = self.bounds[1]
            else:
                raise Exception("If normalization is enabled either provide pf or bounds!")
        else:
            self.ideal_point = np.zeros(self.n_dim)
            self.nadir_point = np.ones(self.n_dim)

        self.range = self.nadir_point - self.ideal_point

        return self._calc(F)

    @abc.abstractmethod
    def _calc(self, F):
        return
