"""Hypervolume indicator for multi-objective optimization."""

from numpy import ndarray

from pymoo.core.indicator import Indicator
from pymoo.indicators.distance_indicator import derive_ideal_and_nadir_from_pf
from pymoo.util.misc import at_least_2d_array
from moocore import hypervolume as _hypervolume


class Hypervolume(Indicator):
    """Hypervolume indicator for solution quality assessment."""

    def __init__(
        self,
        ref_point: ndarray | None = None,
        pf: ndarray | None = None,
        nds: bool = True,
        norm_ref_point: bool = True,
        ideal: ndarray | None = None,
        nadir: ndarray | None = None,
        **kwargs,
    ) -> None:
        """Initialize hypervolume indicator.

        Args:
            ref_point: Reference point for hypervolume calculation.
            pf: Pareto front reference.
            nds: Whether to use non-dominated sorting (deprecated).
            norm_ref_point: Whether to normalize the reference point.
            ideal: Ideal point.
            nadir: Nadir point.
            **kwargs: Additional keyword arguments passed to parent.
        """
        pf = at_least_2d_array(pf, extend_as="row")  # type: ignore[assignment]
        ideal, nadir = derive_ideal_and_nadir_from_pf(pf, ideal=ideal, nadir=nadir)

        super().__init__(ideal=ideal, nadir=nadir, **kwargs)
        # self.normalization = ZeroToOneNormalization(ideal, nadir)

        # whether the input should be checked for domination or not (deprecated)
        self.nds = nds

        # the reference point that shall be used - either derived from pf or provided
        ref_point = ref_point
        if ref_point is None:
            if pf is not None:
                ref_point = pf.max(axis=0)

        # we also have to normalize the reference point to have the same scales
        if norm_ref_point:
            ref_point = self.normalization.forward(ref_point)

        self.ref_point = ref_point
        assert self.ref_point is not None, (
            "For Hypervolume a reference point needs to be provided!"
        )

    def _do(self, F: ndarray) -> float:
        # calculate the hypervolume using moocore
        val = _hypervolume(F, ref=self.ref_point)
        return val


class HV(Hypervolume):
    """Alias for Hypervolume indicator."""

    pass  # noqa: E701
