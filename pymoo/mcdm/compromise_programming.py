"""Compromise programming decision making method."""

from pymoo.core.decision_making import DecisionMaking
from pymoo.util.normalization import normalize


class CompromiseProgramming(DecisionMaking):
    """Compromise programming decision making method."""

    def __init__(self, metric: str = "euclidean", **kwargs) -> None:
        """Initialize compromise programming.

        Args:
            metric: Distance metric for compromise solution.
            **kwargs: Additional arguments for DecisionMaking.
        """
        super().__init__(**kwargs)
        self.metric = metric

    def _do(self, F, **kwargs):
        """Compute compromise solution.

        Args:
            F: Objective values.
            **kwargs: Additional arguments.

        Returns:
            Compromise solution index.
        """
        F, _, ideal_point, nadir_point = normalize(
            F,
            xl=self.ideal_point,
            xu=self.nadir_point,
            estimate_bounds_if_none=True,
            return_bounds=True,
        )

        return None
