import numpy as np

from pymoo.model.indicator import Indicator
from pymoo.util.misc import vectorized_cdist


class DistanceIndicator(Indicator):

    def __init__(self, pf, _type="igd+", **kwargs):
        kwargs['pf'] = pf
        super().__init__(**kwargs)
        self._type = _type

    def _calc(self, F):

        def euclidean_distance(a, b):
            return np.sqrt((((a - b) / self.range) ** 2).sum(axis=1))

        def modified_distance(z, a):
            d = a - z
            d[d < 0] = 0
            d = d / self.range
            return np.sqrt((d ** 2).sum(axis=1))

        if self._type == "gd" or self._type == "igd":
            dist_func = euclidean_distance
        if self._type == "gd+" or self._type == "igd+":
            dist_func = modified_distance

        D = vectorized_cdist(self.pf, F, func_dist=dist_func)

        if self._type == "gd" or self._type == "gd+":
            axis = 0
        if self._type == "igd" or self._type == "igd+":
            axis = 1

        return np.mean(np.min(D, axis=axis))


class GD(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, _type="gd", **kwargs)


class GDPlus(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, _type="gd+", **kwargs)


class IGD(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, _type="igd", **kwargs)


class IGDPlus(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, _type="igd+", **kwargs)
