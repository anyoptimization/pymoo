import numpy as np

from pymoo.core.indicator import Indicator
from pymoo.util.misc import vectorized_cdist, at_least_2d_array


def euclidean_distance(a, b, norm=None):
    return np.sqrt((((a - b) / norm) ** 2).sum(axis=1))


def modified_distance(z, a, norm=None):
    d = a - z
    d[d < 0] = 0
    d = d / norm
    return np.sqrt((d ** 2).sum(axis=1))


def derive_ideal_and_nadir_from_pf(pf, ideal=None, nadir=None):

    # try to derive ideal and nadir if not already set and pf provided
    if pf is not None:
        if ideal is None:
            ideal = np.min(pf, axis=0)
        if nadir is None:
            nadir = np.max(pf, axis=0)

    return ideal, nadir


class DistanceIndicator(Indicator):

    def __init__(self, pf, dist_func, axis, zero_to_one=False, ideal=None, nadir=None, norm_by_dist=False, **kwargs):

        # the pareto front if necessary to calculate the indicator
        pf = at_least_2d_array(pf, extend_as="row")
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(zero_to_one=zero_to_one, ideal=ideal, nadir=nadir, **kwargs)
        self.dist_func = dist_func
        self.axis = axis
        self.norm_by_dist = norm_by_dist
        self.pf = self.normalization.forward(pf)

    def _do(self, F):

        # a factor to normalize the distances by (1.0 disables that by default)
        norm = 1.0

        # if zero_to_one is disabled this can be used to normalize the distance calculation itself
        if self.norm_by_dist:
            assert self.ideal is not None and self.nadir is not None, "If norm_by_dist is enabled ideal and nadir must be set!"
            norm = self.nadir - self.ideal

        D = vectorized_cdist(self.pf, F, func_dist=self.dist_func, norm=norm)
        return np.mean(np.min(D, axis=self.axis))
