"""Rounding repair operator for converting continuous to integer variables."""

import numpy as np

from pymoo.core.repair import Repair


class RoundingRepair(Repair):
    """Repair operator that rounds continuous variables to nearest integers."""

    def __init__(self, **kwargs) -> None:
        """Initialize the rounding repair operator."""
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):  # noqa: D417
        """Round variables to nearest integers.

        Args:
            problem: The optimization problem.
            X: Population variables.

        Returns:
            Rounded integer variables.
        """
        return np.around(X).astype(int)
