import numpy as np

from pymoo.model.indicator import Indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from pymoo.vendor.hv import HyperVolume as _HyperVolume


class Hypervolume(Indicator):

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
