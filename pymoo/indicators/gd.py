"""Generational Distance indicator."""

from numpy import ndarray

from pymoo.indicators.distance_indicator import DistanceIndicator, euclidean_distance


class GD(DistanceIndicator):
    """Generational Distance indicator for solution quality.

    GD measures the average Euclidean distance from each solution in the
    obtained set to the nearest point on the reference Pareto front.
    """

    def __init__(self, pf: ndarray, **kwargs) -> None:
        """Initialize GD indicator.

        Args:
            pf: Pareto front reference.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(pf, euclidean_distance, 0, **kwargs)
