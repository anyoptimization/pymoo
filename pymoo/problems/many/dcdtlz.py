"""Degenerate constrained DTLZ test problems (DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3)."""

import pymoo.gradient.toolbox as anp

from pymoo.problems.many.dtlz import DTLZ1, DTLZ3


def constraint_dc1(X, a: int = 5, b: float = 0.95):
    """Compute DC1 constraint.

    Args:
        X: Decision variables.
        a: Frequency parameter.
        b: Offset parameter.

    Returns:
        Constraint values.
    """
    G = b - anp.cos(a * anp.pi * X[:, 0])
    return G


def constraints_dc2(gx, a: int = 3, b: float = 0.9):
    """Compute DC2 constraints.

    Args:
        gx: gx values from DTLZ.
        a: Frequency parameter.
        b: Offset parameter.

    Returns:
        Constraint values.
    """
    G = anp.column_stack([b - anp.cos(gx / 100 * anp.pi * a), b - anp.exp(-gx / 100)])
    return G


def constraints_dc3(X, gx, a: int = 5, b: float = 0.5):
    """Compute DC3 constraints.

    Args:
        X: Decision variables.
        gx: gx values from DTLZ.
        a: Frequency parameter.
        b: Offset parameter.

    Returns:
        Constraint values.
    """
    Ggx = b - anp.cos(a * anp.pi * gx)
    Gx = b - anp.cos(a * anp.pi * X)
    return anp.column_stack([Ggx, Gx])


class DC1DTLZ1(DTLZ1):
    """Degenerate constrained DTLZ1 problem with one inequality constraint."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, **kwargs) -> None:
        """Initialize DC1DTLZ1.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var, n_obj, **kwargs)
        self.n_ieq_constr = 1

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and constraint.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_dc1(x)


class DC1DTLZ3(DTLZ3):
    """Degenerate constrained DTLZ3 problem with one inequality constraint."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, **kwargs) -> None:
        """Initialize DC1DTLZ3.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var, n_obj, **kwargs)
        self.n_ieq_constr = 1

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and constraint.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_dc1(x)


class DC2DTLZ1(DTLZ1):
    """Degenerate constrained DTLZ1 problem with two inequality constraints."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, **kwargs) -> None:
        """Initialize DC2DTLZ1.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var, n_obj, **kwargs)
        self.n_ieq_constr = 2

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and constraints.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)
        out["G"] = constraints_dc2(g)


class DC2DTLZ3(DTLZ3):
    """Degenerate constrained DTLZ3 problem with two inequality constraints."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, **kwargs) -> None:
        """Initialize DC2DTLZ3.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var, n_obj, **kwargs)
        self.n_ieq_constr = 2

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and constraints.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)
        out["G"] = constraints_dc2(g)


class DC3DTLZ1(DTLZ1):
    """Degenerate constrained DTLZ1 problem with multiple inequality constraints."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, **kwargs) -> None:
        """Initialize DC3DTLZ1.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var, n_obj, n_ieq_constr=n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and constraints.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)
        out["G"] = constraints_dc3(X_, g)


class DC3DTLZ3(DTLZ3):
    """Degenerate constrained DTLZ3 problem with multiple inequality constraints."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, **kwargs) -> None:
        """Initialize DC3DTLZ3.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var, n_obj, n_ieq_constr=n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and constraints.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)
        out["G"] = constraints_dc3(X_, g)
