"""Variable type repair operator."""

from pymoo.core.repair import Repair


class TypeRepair(Repair):
    """Repair operator that converts variables to a specific data type."""

    def __init__(self, vtype) -> None:
        """Initialize the type repair operator.

        Args:
            vtype: Target data type for variables.
        """
        super().__init__()
        self.vtype = vtype

    def _do(self, problem, X, **kwargs):  # noqa: D417
        """Convert variables to the target type.

        Args:
            problem: The optimization problem.
            X: Population variables.

        Returns:
            Variables converted to target type.
        """
        return X.astype(self.vtype)
