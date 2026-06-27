"""Spacing indicator for Pareto front uniformity."""

import numpy as np
from pymoo.core.indicator import Indicator
from pymoo.indicators.distance_indicator import (
    at_least_2d_array,
    derive_ideal_and_nadir_from_pf,
)
from scipy.spatial.distance import pdist, squareform


class SpacingIndicator(Indicator):
    """Spacing indicator for Pareto front uniformity."""

    def __init__(
        self,
        pf=None,
        zero_to_one=False,
        ideal=None,
        nadir=None,
    ):
        """Initialize spacing indicator.

        Args:
            pf: Pareto front.
            zero_to_one: Whether to normalize objective values.
            ideal: Ideal point.
            nadir: Nadir point.
        """
        # the pareto front if necessary to calculate the indicator
        pf = at_least_2d_array(pf, extend_as="row")
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(
            pf=pf,
            zero_to_one=zero_to_one,
            ideal=ideal,
            nadir=nadir,
        )

    def _do(self, F, *args, **kwargs):

        # Get F dimensions
        n_points, n_obj = F.shape

        # knn
        D = squareform(pdist(F, metric="cityblock"))
        d = np.partition(D, 1, axis=1)[:, 1]
        dm = np.mean(d)

        # Get spacing
        S = np.sqrt(np.sum(np.square(d - dm)) / n_points)

        return S
