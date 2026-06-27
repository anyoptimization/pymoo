"""Roulette wheel selection for probability-based selection."""

import numpy as np
from numpy.typing import NDArray

from pymoo.util import default_random_state


class RouletteWheelSelection:
    """Roulette wheel selection based on fitness values."""

    def __init__(self, val: NDArray, larger_is_better: bool = True) -> None:
        """Initialize roulette wheel selection.

        Args:
            val: Array of fitness values.
            larger_is_better: If True, larger values have higher selection probability.
        """
        super().__init__()
        if not larger_is_better:
            val = val.max() - val
        _sum = val.sum()
        self.cumulative = np.array([val[:k].sum() / _sum for k in range(1, len(val))])

    @default_random_state
    def next(
        self,
        n: int | None = None,
        random_state=None,  # type: ignore[misc]
    ) -> int | NDArray:
        """Select individuals using roulette wheel.

        Args:
            n: Number of individuals to select (None for single selection).
            random_state: Random state for reproducibility.

        Returns:
            Index or array of indices of selected individuals.
        """
        if n is None:
            X = random_state.random((1, 1))
        else:
            X = random_state.random((n, 1))
            if n > 1:
                X.repeat(n - 1, axis=1)

        M = self.cumulative[None, :].repeat(len(X), axis=0)
        B = X >= M
        ret = B.sum(axis=1)

        if n is None:
            return int(ret[0])
        return ret
