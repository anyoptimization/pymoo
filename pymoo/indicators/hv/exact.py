import numpy as np
from moocore import hypervolume as hv
from moocore import hv_contributions as hvc


def hv_exact(ref_point, F):
    return hv(F, ref=ref_point)


def hvc_exact(ref_point, F):
    return hvc(F, ref=ref_point)


class DynamicHypervolume:

    def __init__(self, ref_point, F=None, func_hv=None, func_hvc=None) -> None:
        super().__init__()
        self.ref_point = ref_point

        self.func_hv = func_hv
        self.func_hvc = func_hvc

        self.n_dim = len(ref_point)
        self.F = np.zeros((0, self.n_dim))

        self.hv = 0.0
        self.hvc = np.zeros(0)

        if F is not None:
            self.add(F)

    def add(self, F):
        assert len(F.shape) == 2, "The points to add must be a two-dimensional array."
        assert F.shape[1] == self.n_dim, "The dimensions of the ref_point and points to add must be equal"
        self.F = np.vstack([self.F, F])
        self.hv, self.hvc = self.calc()
        return self

    def delete(self, k):
        assert k < len(self.F)
        self.F = np.delete(self.F, k, axis=0)
        self.hv, self.hvc = self.calc()
        return self

    def calc(self):
        return self._calc(self.ref_point, self.F)

        # if len(self.F) == 0:
        #     return 0.0, np.zeros(0)
        # else:
        #     return self._calc(self.ref_point, self.F)

    def _calc(self, ref_point, F):
        hv = None
        if self.func_hv is not None:
            hv = self.func_hv(ref_point, F)

        hvc = None
        if self.func_hvc is not None:
            hvc = self.func_hvc(ref_point, F)

        return hv, hvc


class ExactHypervolume(DynamicHypervolume):

    def __init__(self, ref_point, func_hv=hv_exact, func_hvc=hvc_exact, **kwargs) -> None:
        super().__init__(ref_point, func_hv=func_hv, func_hvc=func_hvc, **kwargs)
