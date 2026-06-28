"""DTLZ test problems for many-objective optimization."""

from typing import Optional

import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.util.remote import Remote


class DTLZ(Problem):
    """Base class for DTLZ test problems.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        k: Number of local Pareto fronts (optional).
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(
        self,
        n_var: Optional[int] = None,
        n_obj: Optional[int] = None,
        k: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize DTLZ problem."""
        if n_var is not None:
            k_val = n_var - n_obj + 1  # type: ignore
            self.k = k_val
            n_var_val = n_var
        elif k is not None:
            self.k = k
            n_var_val = k + n_obj - 1  # type: ignore
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var_val, n_obj=n_obj, xl=0, xu=1, vtype=float, **kwargs)

    def g1(self, X_M):
        """Compute g1 auxiliary function.

        Args:
            X_M: Last k variables.

        Returns:
            Auxiliary function values.
        """
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        """Compute g2 auxiliary function.

        Args:
            X_M: Last k variables.

        Returns:
            Auxiliary function values.
        """
        return anp.sum(anp.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha: int = 1):
        """Compute objective functions.

        Args:
            X_: First (M-1) variables.
            g: Auxiliary function values.
            alpha: Exponent parameter.

        Returns:
            Objective values.
        """
        f = []

        for i in range(0, self.n_obj):
            _f = 1 + g
            _f *= anp.prod(
                anp.cos(anp.power(X_[:, : X_.shape[1] - i], alpha) * anp.pi / 2.0),
                axis=1,
            )
            if i > 0:
                _f *= anp.sin(anp.power(X_[:, X_.shape[1] - i], alpha) * anp.pi / 2.0)

            f.append(_f)

        f = anp.column_stack(f)
        return f


class DTLZ1(DTLZ):
    """DTLZ1 test problem with a simple linear Pareto front.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, n_var: int = 7, n_obj: int = 3, **kwargs) -> None:
        """Initialize DTLZ1."""
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        """Calculate Pareto front.

        Args:
            ref_dirs: Reference directions.

        Returns:
            Pareto front solutions.
        """
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return 0.5 * ref_dirs

    def obj_func(self, X_, g):
        """Compute objective functions for DTLZ1.

        Args:
            X_: First (M-1) variables.
            g: Auxiliary function values.

        Returns:
            Objective values.
        """
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= anp.prod(X_[:, : X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        return anp.column_stack(f)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate DTLZ1 problem.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g)


class DTLZ2(DTLZ):
    """DTLZ2 test problem with a spherical Pareto front.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, n_var: int = 10, n_obj: int = 3, **kwargs) -> None:
        """Initialize DTLZ2."""
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        """Calculate Pareto front.

        Args:
            ref_dirs: Reference directions.

        Returns:
            Pareto front solutions.
        """
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate DTLZ2 problem.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)


class DTLZ3(DTLZ):
    """DTLZ3 test problem with multimodal Pareto front.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, n_var: int = 10, n_obj: int = 3, **kwargs) -> None:
        """Initialize DTLZ3."""
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        """Calculate Pareto front.

        Args:
            ref_dirs: Reference directions.

        Returns:
            Pareto front solutions.
        """
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate DTLZ3 problem.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)


class DTLZ4(DTLZ):
    """DTLZ4 test problem with biased Pareto front.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        alpha: Exponent parameter.
        d: Distance scaling parameter (unused).
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, n_var: int = 10, n_obj: int = 3, alpha: int = 100, d: int = 100, **kwargs) -> None:
        """Initialize DTLZ4."""
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)
        self.alpha = alpha
        self.d = d

    def _calc_pareto_front(self, ref_dirs=None):
        """Calculate Pareto front.

        Args:
            ref_dirs: Reference directions.

        Returns:
            Pareto front solutions.
        """
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate DTLZ4 problem.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=self.alpha)


class DTLZ5(DTLZ):
    """DTLZ5 test problem with a degenerate curved Pareto front.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, n_var: int = 10, n_obj: int = 3, **kwargs) -> None:
        """Initialize DTLZ5."""
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _calc_pareto_front(self, n_points: int = 500):
        """Calculate Pareto front.

        Args:
            n_points: Number of points to generate.

        Returns:
            Pareto front solutions.
        """
        if self.n_obj == 3:
            return Remote.get_instance().load("pymoo", "pf", "dtlz5-3d.pf")

        # Degenerate curve: theta_1 free in [0, pi/2], theta_i = pi/4 for i > 1
        theta1 = np.linspace(0, np.pi / 2, n_points)
        thetas = np.column_stack([theta1] + [np.full(n_points, np.pi / 4)] * (self.n_obj - 2))
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        pf = np.zeros((n_points, self.n_obj))
        for i in range(self.n_obj):
            pf[:, i] = np.prod(cos_t[:, : self.n_obj - 1 - i], axis=1)
            if i > 0:
                pf[:, i] *= sin_t[:, self.n_obj - 1 - i]
        return pf

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate DTLZ5 problem.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = anp.column_stack([x[:, 0], theta[:, 1:]])

        out["F"] = self.obj_func(theta, g)


class DTLZ6(DTLZ):
    """DTLZ6 test problem with a degenerate curved Pareto front (different g function).

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, n_var: int = 10, n_obj: int = 3, **kwargs) -> None:
        """Initialize DTLZ6."""
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _calc_pareto_front(self, n_points: int = 500):
        """Calculate Pareto front.

        Args:
            n_points: Number of points to generate.

        Returns:
            Pareto front solutions.
        """
        if self.n_obj == 3:
            return Remote.get_instance().load("pymoo", "pf", "dtlz6-3d.pf")

        # Same degenerate curve geometry as DTLZ5 (different g function, same Pareto front shape)
        theta1 = np.linspace(0, np.pi / 2, n_points)
        thetas = np.column_stack([theta1] + [np.full(n_points, np.pi / 4)] * (self.n_obj - 2))
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        pf = np.zeros((n_points, self.n_obj))
        for i in range(self.n_obj):
            pf[:, i] = np.prod(cos_t[:, : self.n_obj - 1 - i], axis=1)
            if i > 0:
                pf[:, i] *= sin_t[:, self.n_obj - 1 - i]
        return pf

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate DTLZ6 problem.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = anp.sum(anp.power(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = anp.column_stack([x[:, 0], theta[:, 1:]])

        out["F"] = self.obj_func(theta, g)


class DTLZ7(DTLZ):
    """DTLZ7 test problem with a discontinuous Pareto front.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(self, n_var: int = 10, n_obj: int = 3, **kwargs) -> None:
        """Initialize DTLZ7."""
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _calc_pareto_front(self, n_points: int = 500):
        """Calculate Pareto front.

        Args:
            n_points: Number of points to generate.

        Returns:
            Pareto front solutions.
        """
        if self.n_obj == 3:
            return Remote.get_instance().load("pymoo", "pf", "dtlz7-3d.pf")

        # Sample first (n_obj-1) objectives uniformly; compute last via h with g*=1
        # At the Pareto front: g* = 1, so f_M = (1+g*)*h = 2*h
        # h = n_obj - sum(f_i/(1+g*) * (1+sin(3pi*f_i))) = n_obj - sum(f_i/2*(1+sin(3pi*f_i)))
        # => f_M = 2*n_obj - sum(f_i*(1+sin(3pi*f_i)))
        rng = np.random.default_rng(42)
        F_first = rng.random((n_points * 20, self.n_obj - 1))
        F_last = 2 * self.n_obj - np.sum(F_first * (1 + np.sin(3 * np.pi * F_first)), axis=1)
        valid = F_last >= 0
        return np.column_stack([F_first[valid], F_last[valid]])

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate DTLZ7 problem.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = anp.column_stack(f)

        g = 1 + 9 / self.k * anp.sum(x[:, -self.k :], axis=1)
        h = self.n_obj - anp.sum(f / (1 + g[:, None]) * (1 + anp.sin(3 * anp.pi * f)), axis=1)

        out["F"] = anp.column_stack([f, (1 + g) * h])


class InvertedDTLZ1(DTLZ1):
    """Inverted DTLZ1 problem with flipped objectives."""

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate inverted objectives.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        X_M = x[:, self.n_obj - 1 :]
        g = self.g1(X_M)

        super()._evaluate(x, out, *args, **kwargs)
        out["F"] = 0.5 * (1 + g[:, None]) - out["F"]

    def _calc_pareto_front(self, *args, **kwargs):
        """Calculate Pareto front.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Pareto front solutions.
        """
        return self.problem.pareto_front(*args, **kwargs)


class ScaledProblem(Problem):
    """Wrapper to scale problem objectives by power factors.

    Args:
        problem: Base problem to scale.
        scale_factor: Base scaling factor.
    """

    def __init__(self, problem: Problem, scale_factor: float) -> None:
        """Initialize ScaledProblem."""
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_ieq_constr=problem.n_ieq_constr,
            n_eq_constr=problem.n_eq_constr,
            xl=problem.xl,
            xu=problem.xu,
            vtype=problem.vtype,
        )
        self.problem = problem
        self.scale_factor = scale_factor

    @staticmethod
    def get_scale(n: int, scale_factor: float):
        """Get scaling factors for objectives.

        Args:
            n: Number of objectives.
            scale_factor: Base scaling factor.

        Returns:
            Scaling array.
        """
        return np.power(np.full(n, scale_factor), np.arange(n))

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate scaled objectives.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.problem._evaluate(x, out, *args, **kwargs)
        out["F"] = out["F"] * ScaledProblem.get_scale(self.n_obj, self.scale_factor)

    def _calc_pareto_front(self, *args, **kwargs):
        """Calculate scaled Pareto front.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Scaled Pareto front solutions.
        """
        return self.problem.pareto_front(*args, **kwargs) * ScaledProblem.get_scale(self.n_obj, self.scale_factor)


class ConvexProblem(Problem):
    """Wrapper to apply convex transformation to objectives.

    Args:
        problem: Base problem to transform.
    """

    def __init__(self, problem: Problem) -> None:
        """Initialize ConvexProblem."""
        super().__init__(
            problem.n_var,
            problem.n_obj,
            problem.n_ieq_constr,
            problem.n_eq_constr,
            problem.xl,
            problem.xu,
        )
        self.problem = problem

    def get_power(self, n: int):
        """Get power exponents for convex transformation.

        Args:
            n: Number of objectives.

        Returns:
            Power exponent array.
        """
        p = np.full(n, 4.0)
        p[-1] = 2.0
        return p

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate with convex transformation.

        Args:
            x: Decision variables.
            out: Output dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.problem._evaluate(x, out, **kwargs)
        out["F"] = anp.power(out["F"], self.get_power(self.n_obj))

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        """Calculate Pareto front with convex transformation.

        Args:
            ref_dirs: Reference directions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Transformed Pareto front solutions.
        """
        F = self.problem.pareto_front(ref_dirs)
        return np.power(F, self.get_power(self.n_obj))


class ScaledDTLZ1(ScaledProblem):
    """Scaled version of DTLZ1.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        scale_factor: Scaling factor.
        **kwargs: Additional arguments passed to DTLZ1.
    """

    def __init__(self, n_var: int = 7, n_obj: int = 3, scale_factor: float = 10, **kwargs) -> None:
        """Initialize ScaledDTLZ1."""
        super().__init__(DTLZ1(n_var=n_var, n_obj=n_obj, **kwargs), scale_factor=scale_factor)


class ConvexDTLZ2(ConvexProblem):
    """Convex version of DTLZ2.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        **kwargs: Additional arguments passed to DTLZ2.
    """

    def __init__(self, n_var: int = 10, n_obj: int = 3, **kwargs) -> None:
        """Initialize ConvexDTLZ2."""
        super().__init__(DTLZ2(n_var=n_var, n_obj=n_obj, **kwargs))


class ConvexDTLZ4(ConvexProblem):
    """Convex version of DTLZ4.

    Args:
        n_var: Number of variables.
        n_obj: Number of objectives.
        **kwargs: Additional arguments passed to DTLZ4.
    """

    def __init__(self, n_var: int = 10, n_obj: int = 3, **kwargs) -> None:
        """Initialize ConvexDTLZ4."""
        super().__init__(DTLZ4(n_var=n_var, n_obj=n_obj, **kwargs))


def generic_sphere(ref_dirs):
    """Normalize reference directions to unit sphere.

    Args:
        ref_dirs: Reference direction vectors.

    Returns:
        Normalized reference directions.
    """
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))


def get_ref_dirs(n_obj: int):
    """Get predefined reference directions for DTLZ problems.

    Args:
        n_obj: Number of objectives.

    Returns:
        Reference direction vectors.

    Raises:
        Exception: If n_obj > 3.
    """
    if n_obj == 2:
        ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()
    elif n_obj == 3:
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=15).do()
    else:
        raise Exception("Please provide reference directions for more than 3 objectives!")
    return ref_dirs
