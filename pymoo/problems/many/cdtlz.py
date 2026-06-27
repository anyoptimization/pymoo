"""Constrained DTLZ test problems (C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4)."""

import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.problems.many.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, get_ref_dirs


class C1DTLZ1(DTLZ1):
    """Constrained DTLZ1 problem with one inequality constraint."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, **kwargs) -> None:
        """Initialize C1DTLZ1.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and linear constraint.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_c1_linear(out["F"])

    def _calc_pareto_front(self, ref_dirs=None, *args, **kwargs):
        """Calculate Pareto front.

        Args:
            ref_dirs: Reference directions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Pareto front solutions.
        """
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return super()._calc_pareto_front(ref_dirs, *args, **kwargs)


class C1DTLZ3(DTLZ3):
    """Constrained DTLZ3 problem with one spherical inequality constraint."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, r=None, **kwargs) -> None:
        """Initialize C1DTLZ3.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            r: Radius of spherical constraint.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, **kwargs)

        if r is None:
            if self.n_obj < 5:
                r = 9.0
            elif 5 <= self.n_obj <= 12:
                r = 12.5
            else:
                r = 15.0

        self.r = r

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and spherical constraint.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_c1_spherical(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs=None, *args, **kwargs):
        """Calculate Pareto front.

        Args:
            ref_dirs: Reference directions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Pareto front solutions.
        """
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return super()._calc_pareto_front(ref_dirs, *args, **kwargs)


class C2DTLZ2(DTLZ2):
    """Constrained DTLZ2 problem with one inequality constraint."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, r=None, **kwargs) -> None:
        """Initialize C2DTLZ2.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            r: Constraint radius parameter.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=1, **kwargs)

        if r is None:
            if n_obj == 2:
                r = 0.2
            elif n_obj == 3:
                r = 0.4
            else:
                r = 0.5

        self.r = r

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and constraint.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_c2(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs=None, *args, **kwargs):
        """Calculate Pareto front.

        Args:
            ref_dirs: Reference directions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Pareto front solutions.
        """
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        F = super()._calc_pareto_front(ref_dirs, *args, **kwargs)
        G = constraint_c2(F, r=self.r)
        G[G <= 0] = 0
        return F[G <= 0]


class C3DTLZ1(DTLZ1):
    """Constrained DTLZ1 problem with multiple linear inequality constraints."""

    def __init__(self, n_var: int = 12, n_obj: int = 3, **kwargs) -> None:
        """Initialize C3DTLZ1.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and linear constraints.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_c3_linear(out["F"])


class C3DTLZ4(DTLZ4):
    """Constrained DTLZ4 problem with multiple spherical inequality constraints."""

    def __init__(self, n_var: int = 7, n_obj: int = 3, **kwargs) -> None:
        """Initialize C3DTLZ4.

        Args:
            n_var: Number of variables.
            n_obj: Number of objectives.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate objectives and spherical constraints.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super()._evaluate(x, out, *args, **kwargs)
        out["G"] = constraint_c3_spherical(out["F"])

    def _calc_pareto_front(self, ref_dirs=None, *args, **kwargs):
        """Calculate Pareto front.

        Args:
            ref_dirs: Reference directions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Pareto front solutions.
        """
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        F = super()._calc_pareto_front(ref_dirs, *args, **kwargs)
        a = np.sqrt(np.sum(F**2, 1) - 3 / 4 * np.max(F**2, axis=1))
        a = np.expand_dims(a, axis=1)
        a = np.tile(a, [1, ref_dirs.shape[1]])
        F = F / a

        return F


def constraint_c1_linear(f):
    """Compute linear constraint for C1DTLZ.

    Args:
        f: Objective values.

    Returns:
        Constraint values.
    """
    g = -(1 - f[:, -1] / 0.6 - anp.sum(f[:, :-1] / 0.5, axis=1))
    return g


def constraint_c1_spherical(f, r):
    """Compute spherical constraint for C1DTLZ3.

    Args:
        f: Objective values.
        r: Radius parameter.

    Returns:
        Constraint values.
    """
    radius = anp.sum(f**2, axis=1)
    g = -(radius - 16) * (radius - r**2)

    return g


def constraint_c2(f, r):
    """Compute constraint for C2DTLZ2.

    Args:
        f: Objective values.
        r: Constraint radius parameter.

    Returns:
        Constraint values.
    """
    n_obj = f.shape[1]

    v1 = anp.full(f.shape[0], anp.inf)

    for i in range(n_obj):
        temp = (f[:, i] - 1) ** 2 + (anp.sum(f**2, axis=1) - f[:, i] ** 2) - r**2
        v1 = anp.minimum(temp.flatten(), v1)

    a = 1 / anp.sqrt(n_obj)
    v2 = anp.sum((f - a) ** 2, axis=1) - r**2
    g = anp.minimum(v1, v2.flatten())

    return g


def constraint_c3_linear(f):
    """Compute multiple linear constraints for C3DTLZ1.

    Args:
        f: Objective values.

    Returns:
        Constraint values.
    """
    n_obj = f.shape[1]
    g = []

    for i in range(n_obj):
        _g = 1 - f[:, i] / 0.5 - (anp.sum(f, axis=1) - f[:, i])
        g.append(_g)

    return anp.column_stack(g)


def constraint_c3_spherical(f):
    """Compute multiple spherical constraints for C3DTLZ4.

    Args:
        f: Objective values.

    Returns:
        Constraint values.
    """
    n_obj = f.shape[1]
    g = []

    for i in range(n_obj):
        _g = 1 - f[:, i] ** 2 / 4 - (anp.sum(f**2, axis=1) - f[:, i] ** 2)
        g.append(_g)
    return anp.column_stack(g)


def constraint_c4_cylindrical(f, r):
    """Compute cylindrical constraint.

    Args:
        f: Objective values.
        r: Radius parameter.

    Returns:
        Constraint values.
    """
    l = anp.mean(f, axis=1)  # noqa: E741
    l = anp.expand_dims(l, axis=1)  # noqa: E741
    g = -anp.sum(anp.power(f - l, 2), axis=1) + anp.power(r, 2)
    return g
