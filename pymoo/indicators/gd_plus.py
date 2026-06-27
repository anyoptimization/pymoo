"""Generational Distance Plus indicator."""

from numpy import ndarray

from pymoo.indicators.distance_indicator import DistanceIndicator, modified_distance


class GDPlus(DistanceIndicator):
    """Generational Distance Plus indicator for solution quality.

    GD+ measures the average distance from each solution in the obtained set
    to the nearest point on the reference Pareto front (modified distance).
    """

    def __init__(self, pf: ndarray, **kwargs) -> None:
        """Initialize GD+ indicator.

        Args:
            pf: Pareto front reference.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(pf, modified_distance, 0, **kwargs)
