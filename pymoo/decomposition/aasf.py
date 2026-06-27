"""Augmented Achievement Scalarization Function for decomposition-based MOO."""

from typing import Any, Optional

import numpy as np

from pymoo.decomposition.asf import ASF


class AASF(ASF):
    """Augmented Achievement Scalarization Function."""

    def __init__(
        self,
        eps: float = 1e-10,
        _type: str = "auto",
        rho: Optional[float] = None,
        beta: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AASF.

        Args:
            eps: Small epsilon value.
            _type: Type of decomposition.
            rho: Augmentation parameter.
            beta: Alternative parameter to compute rho.
            kwargs: Additional arguments for parent class.

        Raises:
            Exception: If neither rho nor beta is provided.
        """
        super().__init__(eps, _type, **kwargs)
        if rho is None and beta is None:
            raise Exception("Either provide rho or beta!")  # noqa: EM101
        elif rho is None:
            self.rho = calc_rho(beta)
        else:
            self.rho = rho

    def _do(self, F: Any, weights: Any, weight_0: float = 1e-10, **kwargs: Any) -> Any:
        """Compute AASF values.

        Args:
            F: Objective values.
            weights: Decomposition weights.
            weight_0: Minimum weight threshold.
            kwargs: Additional arguments.

        Returns:
            AASF scalarization values.
        """
        asf = super()._do(F, weights, weight_0=weight_0, **kwargs)
        augment = ((F - self.utopian_point) / np.clip(weights, 1e-12, np.inf)).sum(
            axis=1
        )
        return asf + self.rho * augment


def calc_rho(beta: float) -> float:
    """Calculate augmentation parameter from angle.

    Args:
        beta: Angle parameter.

    Returns:
        Augmentation parameter rho.
    """
    return 1 / (1 - np.tan(beta / 360 * 2 * np.pi)) - 1
