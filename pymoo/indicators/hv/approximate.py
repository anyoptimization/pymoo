"""Approximate hypervolume calculation."""

import numpy as np
from numpy import ndarray

from moocore import hv_approx, hv_contributions


class ApproximateHypervolume:
    """Approximate hypervolume calculator.

    Supports incremental addition and deletion of solutions with
    approximate hypervolume and contribution calculations.
    """

    def __init__(
        self,
        ref_point: ndarray,
        n_samples: int = 10000,
        method: str = "Rphi-FWE+",
        seed: int | None = None,
    ) -> None:
        """Initialize approximate hypervolume calculator.

        Args:
            ref_point: Reference point for hypervolume calculation.
            n_samples: Number of samples for approximation.
            method: Approximation method name.
            seed: Random seed for reproducibility.
        """
        self.ref_point = ref_point
        self.n_samples = n_samples
        self.method = method
        self.seed = seed
        self.F: ndarray = np.zeros((0, len(ref_point)))
        self.hv: float = 0.0
        self.hvc: ndarray = np.zeros(0)

    def add(self, F: ndarray) -> "ApproximateHypervolume":
        """Add solutions to the approximation.

        Args:
            F: Solutions to add.

        Returns:
            Self for method chaining.
        """
        self.F = np.vstack([self.F, F])
        self._update()
        return self

    def delete(self, k: int) -> "ApproximateHypervolume":
        """Delete a solution by index.

        Args:
            k: Index of solution to delete.

        Returns:
            Self for method chaining.
        """
        self.F = np.delete(self.F, k, axis=0)
        self._update()
        return self

    def _update(self) -> None:
        """Update hypervolume and contributions."""
        if len(self.F) == 0:
            self.hv = 0.0
            self.hvc = np.zeros(0)
        else:
            self.hv = hv_approx(
                self.F,
                ref=self.ref_point,
                nsamples=self.n_samples,
                method=self.method,
                seed=self.seed,
            )
            self.hvc = hv_contributions(self.F, ref=self.ref_point)
