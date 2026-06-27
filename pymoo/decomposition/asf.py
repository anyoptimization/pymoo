"""Achievement Scalarization Function for decomposition-based MOO."""

from typing import Any

from pymoo.core.decomposition import Decomposition


class ASF(Decomposition):
    """Achievement Scalarization Function."""

    def _do(self, F: Any, weights: Any, weight_0: float = 1e-10, **kwargs: Any) -> Any:
        """Compute ASF values.

        Args:
            F: Objective values.
            weights: Decomposition weights.
            weight_0: Minimum weight threshold.
            kwargs: Additional arguments.

        Returns:
            ASF scalarization values.
        """
        _weights = weights.astype(float)
        _weights[weights == 0] = weight_0  # noqa: E501
        asf = ((F - self.utopian_point) / _weights).max(axis=1)
        return asf
