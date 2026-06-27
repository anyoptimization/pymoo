"""Inverted Generational Distance Plus indicator."""

from numpy import ndarray

from moocore import igd_plus as _igd_plus

from pymoo.indicators.distance_indicator import DistanceIndicator, modified_distance


class IGDPlus(DistanceIndicator):
    """Inverted Generational Distance Plus indicator.

    IGD+ measures the average distance from each reference point to the
    nearest solution in the obtained set (inverse of GD+).
    """

    def __init__(self, pf: ndarray, **kwargs) -> None:
        """Initialize IGD+ indicator.

        Args:
            pf: Pareto front reference.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(pf, modified_distance, 1, **kwargs)

    def _do(self, F: ndarray) -> float:
        if self.norm_by_dist:
            return super()._do(F)
        return _igd_plus(F, ref=self.pf)
