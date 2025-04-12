import numpy as np

from pymoo.core.indicator import Indicator
from pymoo.indicators.distance_indicator import derive_ideal_and_nadir_from_pf
from pymoo.util.misc import at_least_2d_array
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.vendor.hv import HyperVolume as _HyperVolume


class Hypervolume(Indicator):

    def __init__(self, ref_point=None, pf=None, nds=True, norm_ref_point=True, ideal=None, nadir=None, **kwargs):

        pf = at_least_2d_array(pf, extend_as="row")
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(ideal=ideal, nadir=nadir, **kwargs)
        # self.normalization = ZeroToOneNormalization(ideal, nadir)

        # whether the input should be checked for domination or not
        self.nds = nds

        # the reference point that shall be used - either derived from pf or provided
        ref_point = ref_point
        if ref_point is None:
            if pf is not None:
                ref_point = pf.max(axis=0)

        # we also have to normalize the reference point to have the same scales
        if norm_ref_point:
            ref_point = self.normalization.forward(ref_point)

        self.ref_point = ref_point
        assert self.ref_point is not None, "For Hypervolume a reference point needs to be provided!"

    def _do(self, F):
        if self.nds:
            non_dom = NonDominatedSorting().do(F, only_non_dominated_front=True)
            F = np.copy(F[non_dom, :])

        # calculate the hypervolume using a vendor library
        hv = _HyperVolume(self.ref_point)
        val = hv.compute(F)
        return val


class HV(Hypervolume):
    pass


def hvc_looped(ref_point, F, func):
    hv = func(ref_point, F)

    hvc = []

    for k in range(len(F)):
        v = np.full(len(F), True)
        v[k] = False
        _hv = func(ref_point, F[v])
        hvc.append(hv - _hv)

    hvc = np.array(hvc)
    return hvc
