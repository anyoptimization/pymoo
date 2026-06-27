"""Inverted Generational Distance (IGD) indicator."""

from moocore import igd as _igd

from pymoo.indicators.distance_indicator import DistanceIndicator, euclidean_distance


class IGD(DistanceIndicator):
    """Inverted Generational Distance (IGD) indicator."""

    def __init__(self, pf, **kwargs):
        super().__init__(pf, euclidean_distance, 1, **kwargs)

    def _do(self, F):
        if self.norm_by_dist:
            return super()._do(F)
        return _igd(F, ref=self.pf)
