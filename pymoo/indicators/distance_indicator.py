"""Distance-based indicators for multi-objective optimization."""

from typing import Callable

import numpy as np
from numpy import ndarray

from pymoo.core.indicator import Indicator
from pymoo.util.misc import vectorized_cdist, at_least_2d_array


def euclidean_distance(a: ndarray, b: ndarray, norm: ndarray | float | None = None) -> ndarray:
    """Calculate Euclidean distance between points."""
    return np.sqrt((((a - b) / norm) ** 2).sum(axis=1))


def modified_distance(z: ndarray, a: ndarray, norm: ndarray | float | None = None) -> ndarray:
    """Calculate modified distance (truncated below zero)."""
    d = a - z
    d[d < 0] = 0
    d = d / norm
    return np.sqrt((d**2).sum(axis=1))


def derive_ideal_and_nadir_from_pf(
    pf: ndarray | None,
    ideal: ndarray | None = None,
    nadir: ndarray | None = None,
) -> tuple[ndarray | None, ndarray | None]:
    """Derive ideal and nadir points from Pareto front if not provided."""
    # try to derive ideal and nadir if not already set and pf provided
    if pf is not None:
        if ideal is None:
            ideal = np.min(pf, axis=0)
        if nadir is None:
            nadir = np.max(pf, axis=0)

    return ideal, nadir


class DistanceIndicator(Indicator):
    """Distance-based indicator for measuring solution quality."""

    def __init__(
        self,
        pf: ndarray,
        dist_func: Callable,
        axis: int,
        zero_to_one: bool = False,
        ideal: ndarray | None = None,
        nadir: ndarray | None = None,
        norm_by_dist: bool = False,
        **kwargs,
    ):
        """Initialize distance indicator.

        Args:
            pf: Pareto front reference.
            dist_func: Distance function to use.
            axis: Axis along which to compute distances.
            zero_to_one: Whether to normalize to [0, 1].
            ideal: Ideal point.
            nadir: Nadir point.
            norm_by_dist: Whether to normalize by distance range.
            **kwargs: Additional keyword arguments passed to parent.
        """
        # the pareto front if necessary to calculate the indicator
        pf = at_least_2d_array(pf, extend_as="row")  # type: ignore[assignment]
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(zero_to_one=zero_to_one, ideal=ideal, nadir=nadir, **kwargs)
        self.dist_func = dist_func
        self.axis = axis
        self.norm_by_dist = norm_by_dist
        self.pf = self.normalization.forward(pf)

    def _do(self, F: ndarray) -> float:
        # a factor to normalize the distances by (1.0 disables that by default)
        norm = 1.0

        # if zero_to_one is disabled this can be used to normalize the distance calculation itself
        if self.norm_by_dist:
            assert self.ideal is not None and self.nadir is not None, (
                "If norm_by_dist is enabled ideal and nadir must be set!"
            )
            norm = self.nadir - self.ideal

        D = vectorized_cdist(self.pf, F, func_dist=self.dist_func, norm=norm)
        return float(np.mean(np.min(D, axis=self.axis)))  # type: ignore[call-overload]
