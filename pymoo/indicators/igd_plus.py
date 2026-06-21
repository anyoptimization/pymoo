from moocore import igd_plus as _igd_plus

from pymoo.indicators.distance_indicator import DistanceIndicator, modified_distance


class IGDPlus(DistanceIndicator):

    def __init__(self, pf, **kwargs):
        super().__init__(pf, modified_distance, 1, **kwargs)

    def _do(self, F):
        if self.norm_by_dist:
            return super()._do(F)
        return _igd_plus(F, ref=self.pf)
