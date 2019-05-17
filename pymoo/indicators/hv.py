import numpy as np

from pymoo.model.indicator import Indicator
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from pymoo.vendor.hv import HyperVolume as _HyperVolume


class Hypervolume(Indicator):
    def __init__(self, ref_point=None, normalize=False, ideal_point=None, nadir_point=None, pf=None):
        Indicator.__init__(self)
        self.pf = pf
        self.ref_point = ref_point
        self.normalize = normalize
        self.ideal_point = ideal_point
        self.nadir_point = nadir_point

        if pf is not None and ref_point is not None:
            raise Exception("Either provide a reference point or the pareto-front")

        no_boundary_points = (self.ideal_point is None or self.nadir_point is None)
        if (normalize and no_boundary_points) or (normalize and pf is None):
            raise Exception("For the purpose of normalization provide either pf or ideal and nadir point!")

        if pf is not None:
            self.normalize = True
            self.ideal_point = np.min(self.pf, axis=0)
            self.nadir_point = np.max(self.pf, axis=0)

    def _calc(self, F):

        # only consider the non-dominated solutions for HV
        non_dom = NonDominatedSorting().do(F, only_non_dominated_front=True)
        _F = np.copy(F[non_dom, :])

        if self.normalize:
            # because we normalize now the reference point is (1,...1)
            ref_point = np.ones(F.shape[1])
            hv = _HyperVolume(ref_point)
            _F = normalize(_F, x_min=self.ideal_point, x_max=self.nadir_point)
        else:
            hv = _HyperVolume(self.ref_point)

        val = hv.compute(_F)
        return val