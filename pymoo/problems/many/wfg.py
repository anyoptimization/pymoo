"""WFG test problems for many-objective optimization."""

from typing import Optional

import numpy as np

from pymoo.core.problem import Problem
from pymoo.problems.many import generic_sphere, get_ref_dirs
from pymoo.functions import load_function
from pymoo.util.misc import powerset
from pymoo.util import default_random_state


class WFG(Problem):
    """Base class for WFG test problems.

    Implements the WFG (Walking Fish Group) family of many-objective test problems
    with configurable position (k) and distance (l) parameters.
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        k: Optional[int] = None,
        l: Optional[int] = None,  # noqa: E741
        **kwargs,
    ):
        """Initialize a WFG problem instance.

        Args:
            n_var: Number of decision variables.
            n_obj: Number of objectives.
            k: Position parameter; defaults to 4 for 2 objectives, 2*(n_obj-1) otherwise.
            l: Distance parameter; defaults to n_var - k.
            **kwargs: Additional keyword arguments passed to Problem base class.
        """
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=0.0,
            xu=2 * np.arange(1, n_var + 1).astype(float),
            vtype=float,
            **kwargs,
        )

        self.S = np.arange(2, 2 * self.n_obj + 1, 2).astype(float)
        self.A = np.ones(self.n_obj - 1)

        if k:
            self.k = k
        else:
            if n_obj == 2:
                self.k = 4
            else:
                self.k = 2 * (n_obj - 1)

        if l:
            self.l = l
        else:
            self.l = n_var - self.k

        self.validate(self.l, self.k, self.n_obj)

    def validate(self, l: int, k: int, n_obj: int) -> None:  # noqa: E741
        """Validate WFG problem parameters.

        Args:
            l: Distance parameter.
            k: Position parameter.
            n_obj: Number of objectives.

        Raises:
            ValueError: If parameters violate WFG constraints.
        """
        if n_obj < 2:
            raise ValueError("WFG problems must have two or more objectives.")
        if not k % (n_obj - 1) == 0:
            raise ValueError(
                "Position parameter (k) must be divisible by number of objectives minus one."
            )
        if k < 4:
            raise ValueError("Position parameter (k) must be greater or equal than 4.")
        if (k + l) < n_obj:
            raise ValueError(
                "Sum of distance and position parameters must be greater than num. of objs. (k + l >= M)."
            )

    def _post(self, t, a):
        """Post-process transformed variables."""
        x = []
        for i in range(t.shape[1] - 1):
            x.append(np.maximum(t[:, -1], a[i]) * (t[:, i] - 0.5) + 0.5)
        x.append(t[:, -1])
        return np.column_stack(x)

    def _calculate(self, x, s, h):
        """Calculate objective values from post-processed variables."""
        return x[:, -1][:, None] + s * np.column_stack(h)

    @default_random_state
    def _rand_optimal_position(self, n, random_state=None):
        """Generate random optimal positions in the parameter space."""
        return random_state.random((n, self.k))

    def _positional_to_optimal(self, K):
        """Convert positional parameters to optimal decision variables."""
        suffix = np.full((len(K), self.l), 0.35)  # noqa: E741
        X = np.column_stack([K, suffix])
        return X * self.xu

    def _calc_pareto_set_extremes(self):
        """Calculate Pareto set extremes."""
        ps = np.ones((2**self.k, self.k))
        for i, s in enumerate(powerset(np.arange(self.k))):
            ps[i, s] = 0
        return self._positional_to_optimal(ps)

    def _calc_pareto_set_interior(self, n_points):
        """Calculate interior points of the Pareto set."""
        return self._positional_to_optimal(self._rand_optimal_position(n_points))

    def _calc_pareto_set(self, n_points=500, *args, **kwargs):
        """Calculate the Pareto set with extremes and interior points."""
        extremes = self._calc_pareto_set_extremes()
        interior = self._calc_pareto_set_interior(n_points - len(extremes))
        return np.vstack([extremes, interior])

    def _calc_pareto_front(
        self,
        ref_dirs=None,
        n_iterations=200,
        points_each_iteration=200,
        *args,
        **kwargs,
    ):
        """Calculate the Pareto front using reference directions."""
        pf = self.evaluate(self._calc_pareto_set_extremes(), return_values_of=["F"])

        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)

        for k in range(n_iterations):
            _pf = self.evaluate(
                self._calc_pareto_set_interior(points_each_iteration),
                return_values_of=["F"],
            )
            pf = np.vstack([pf, _pf])

            ideal, nadir = pf.min(axis=0), pf.max(axis=0)

            N = (pf - ideal) / (nadir - ideal)
            dist_matrix = load_function("calc_perpendicular_distance")(N, ref_dirs)

            closest = np.argmin(dist_matrix, axis=0)
            pf = pf[closest]

        pf = pf[np.lexsort(pf.T[::-1])]
        return pf


class WFG1(WFG):
    """WFG1 test problem with convex and mixed shapes."""

    @staticmethod
    def t1(x, n, k):
        """First transformation: shift linear."""
        x[:, k:n] = _transformation_shift_linear(x[:, k:n], 0.35)
        return x

    @staticmethod
    def t2(x, n, k):
        """Second transformation: bias flat."""
        x[:, k:n] = _transformation_bias_flat(x[:, k:n], 0.8, 0.75, 0.85)
        return x

    @staticmethod
    def t3(x, n):
        """Third transformation: bias polynomial."""
        x[:, :n] = _transformation_bias_poly(x[:, :n], 0.02)
        return x

    @staticmethod
    def t4(x, m, n, k):
        """Fourth transformation: weighted sum reduction."""
        w = np.arange(2, 2 * n + 1, 2)
        gap = k // (m - 1)
        t = []
        for m in range(1, m):
            _y = x[:, (m - 1) * gap : (m * gap)]
            _w = w[(m - 1) * gap : (m * gap)]
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(x[:, k:n], w[k:n]))
        return np.column_stack(t)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG1 problem."""
        y = x / self.xu
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG1.t2(y, self.n_var, self.k)
        y = WFG1.t3(y, self.n_var)
        y = WFG1.t4(y, self.n_obj, self.n_var, self.k)

        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_mixed(y[:, 0], alpha=1.0, A=5.0))

        out["F"] = self._calculate(y, self.S, h)

    @default_random_state
    def _rand_optimal_position(self, n, random_state=None):
        """Generate random optimal positions with power transformation."""
        return np.power(random_state.random((n, self.k)), 50.0)


class WFG2(WFG):
    """WFG2 test problem with non-separable transformation."""

    def validate(self, l: int, k: int, n_obj: int) -> None:  # noqa: E741
        """Validate WFG2-specific constraints.

        Args:
            l: Distance parameter.
            k: Position parameter.
            n_obj: Number of objectives.
        """
        super().validate(l, k, n_obj)
        validate_wfg2_wfg3(l)

    @staticmethod
    def t2(x, n, k):
        """Second transformation: non-separable reduction."""
        y = [x[:, i] for i in range(k)]

        l = n - k  # noqa: E741
        ind_non_sep = k + l // 2

        i = k + 1
        while i <= ind_non_sep:
            head = k + 2 * (i - k) - 2
            tail = k + 2 * (i - k)
            y.append(_reduction_non_sep(x[:, head:tail], 2))
            i += 1

        return np.column_stack(y)

    @staticmethod
    def t3(x, m, n, k):
        """Third transformation: weighted sum reduction."""
        ind_r_sum = k + (n - k) // 2
        gap = k // (m - 1)

        t = [
            _reduction_weighted_sum_uniform(x[:, (m - 1) * gap : (m * gap)])
            for m in range(1, m)
        ]
        t.append(_reduction_weighted_sum_uniform(x[:, k:ind_r_sum]))

        return np.column_stack(t)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG2 problem."""
        y = x / self.xu
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG2.t2(y, self.n_var, self.k)
        y = WFG2.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_disconnected(y[:, 0], alpha=1.0, beta=1.0, A=5.0))

        out["F"] = self._calculate(y, self.S, h)


class WFG3(WFG):
    """WFG3 test problem with linear shape and degeneracy."""

    def __init__(self, n_var: int, n_obj: int, k: Optional[int] = None, **kwargs):
        """Initialize WFG3 with modified asymmetry vector.

        Args:
            n_var: Number of decision variables.
            n_obj: Number of objectives.
            k: Position parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(n_var, n_obj, k=k, **kwargs)
        self.A[1:] = 0

    def validate(self, l: int, k: int, n_obj: int) -> None:  # noqa: E741
        """Validate WFG3-specific constraints.

        Args:
            l: Distance parameter.
            k: Position parameter.
            n_obj: Number of objectives.
        """
        super().validate(l, k, n_obj)
        validate_wfg2_wfg3(l)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG3 problem."""
        y = x / self.xu
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG2.t2(y, self.n_var, self.k)
        y = WFG2.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_linear(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out["F"] = self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return ref_dirs * self.S


class WFG4(WFG):
    """WFG4 test problem with multi-modal transformation."""

    @staticmethod
    def t1(x):
        """First transformation: multi-modal shift."""
        return _transformation_shift_multi_modal(x, 30.0, 10.0, 0.35)

    @staticmethod
    def t2(x, m, k):
        """Second transformation: weighted sum reduction."""
        gap = k // (m - 1)
        t = [
            _reduction_weighted_sum_uniform(x[:, (m - 1) * gap : (m * gap)])
            for m in range(1, m)
        ]
        t.append(_reduction_weighted_sum_uniform(x[:, k:]))
        return np.column_stack(t)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG4 problem."""
        y = x / self.xu
        y = WFG4.t1(y)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out["F"] = self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG5(WFG):
    """WFG5 test problem with deceptive transformation."""

    @staticmethod
    def t1(x):
        """First transformation: deceptive."""
        return _transformation_param_deceptive(x, A=0.35, B=0.001, C=0.05)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG5 problem."""
        y = x / self.xu
        y = WFG5.t1(y)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out["F"] = self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG6(WFG):
    """WFG6 test problem with non-separable reduction."""

    @staticmethod
    def t2(x, m, n, k):
        """Second transformation: non-separable reduction."""
        gap = k // (m - 1)
        t = [
            _reduction_non_sep(x[:, (m - 1) * gap : (m * gap)], gap)
            for m in range(1, m)
        ]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG6 problem."""
        y = x / self.xu
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG6.t2(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out["F"] = self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG7(WFG):
    """WFG7 test problem with parameter-dependent transformation."""

    @staticmethod
    def t1(x, k):
        """First transformation: parameter-dependent."""
        for i in range(k):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1 :])
            x[:, i] = _transformation_param_dependent(x[:, i], aux)
        return x

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG7 problem."""
        y = x / self.xu
        y = WFG7.t1(y, self.k)
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out["F"] = self._calculate(y, self.S, h)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG8(WFG):
    """WFG8 test problem with cumulative parameter-dependent transformation."""

    @staticmethod
    def t1(x, n, k):
        """First transformation: cumulative parameter-dependent."""
        ret = []
        for i in range(k, n):
            aux = _reduction_weighted_sum_uniform(x[:, :i])
            ret.append(
                _transformation_param_dependent(
                    x[:, i], aux, A=0.98 / 49.98, B=0.02, C=50.0
                )
            )
        return np.column_stack(ret)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG8 problem."""
        y = x / self.xu
        y[:, self.k : self.n_var] = WFG8.t1(y, self.n_var, self.k)
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out["F"] = self._calculate(y, self.S, h)

    def _positional_to_optimal(self, K):
        """Convert positional parameters to optimal decision variables with special handling."""
        k, l = self.k, self.l  # noqa: E741

        for i in range(k, k + l):
            u = K.sum(axis=1) / K.shape[1]
            tmp1 = np.abs(np.floor(0.5 - u) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1.0 - 2.0 * u) * tmp1)
            suffix = np.power(0.35, np.power(tmp2, -1.0))

            K = np.column_stack([K, suffix[:, None]])

        ret = K * (2 * (np.arange(self.n_var) + 1))
        return ret


class WFG9(WFG):
    """WFG9 test problem with mixed transformations."""

    @staticmethod
    def t1(x, n):
        """First transformation: parameter-dependent."""
        ret = []
        for i in range(0, n - 1):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1 :])
            ret.append(_transformation_param_dependent(x[:, i], aux))
        return np.column_stack(ret)

    @staticmethod
    def t2(x, n, k):
        """Second transformation: deceptive and multi-modal."""
        a = [
            _transformation_shift_deceptive(x[:, i], 0.35, 0.001, 0.05)
            for i in range(k)
        ]
        b = [
            _transformation_shift_multi_modal(x[:, i], 30.0, 95.0, 0.35)
            for i in range(k, n)
        ]
        return np.column_stack(a + b)

    @staticmethod
    def t3(x, m, n, k):
        """Third transformation: non-separable reduction."""
        gap = k // (m - 1)
        t = [
            _reduction_non_sep(x[:, (m - 1) * gap : (m * gap)], gap)
            for m in range(1, m)
        ]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the WFG9 problem."""
        y = x / self.xu
        y[:, : self.n_var - 1] = WFG9.t1(y, self.n_var)
        y = WFG9.t2(y, self.n_var, self.k)
        y = WFG9.t3(y, self.n_obj, self.n_var, self.k)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out["F"] = self._calculate(y, self.S, h)

    def _positional_to_optimal(self, K):
        """Convert positional parameters to optimal decision variables with backward recursion."""
        k, l = self.k, self.l  # noqa: E741

        suffix = np.full((len(K), self.l), 0.0)
        X = np.column_stack([K, suffix])
        X[:, self.k + self.l - 1] = 0.35

        for i in range(self.k + self.l - 2, self.k - 1, -1):
            m = X[:, i + 1 : k + l]
            val = m.sum(axis=1) / m.shape[1]
            X[:, i] = 0.35 ** ((0.02 + 1.96 * val) ** -1)

        ret = X * (2 * (np.arange(self.n_var) + 1))
        return ret

    def _calc_pareto_front(self, ref_dirs=None):
        """Calculate the Pareto front using generic sphere."""
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return generic_sphere(ref_dirs) * self.S


# ---------------------------------------------------------------------------------------------------------
# TRANSFORMATIONS
# ---------------------------------------------------------------------------------------------------------


def _transformation_shift_linear(value, shift=0.35):
    """Apply linear shift transformation."""
    return correct_to_01(
        np.fabs(value - shift) / np.fabs(np.floor(shift - value) + shift)
    )


def _transformation_shift_deceptive(y, A=0.35, B=0.005, C=0.05):
    """Apply deceptive shift transformation."""
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


def _transformation_shift_multi_modal(y, A, B, C):
    """Apply multi-modal shift transformation."""
    tmp1 = np.fabs(y - C) / (2.0 * (np.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
    ret = (1.0 + np.cos(tmp2) + 4.0 * B * np.power(tmp1, 2.0)) / (B + 2.0)
    return correct_to_01(ret)


def _transformation_bias_flat(y, a, b, c):
    """Apply flat bias transformation."""
    ret = (
        a
        + np.minimum(0, np.floor(y - b)) * (a * (b - y) / b)
        - np.minimum(0, np.floor(c - y)) * ((1.0 - a) * (y - c) / (1.0 - c))
    )
    return correct_to_01(ret)


def _transformation_bias_poly(y, alpha):
    """Apply polynomial bias transformation."""
    return correct_to_01(y**alpha)


def _transformation_param_dependent(y, y_deg, A=0.98 / 49.98, B=0.02, C=50.0):
    """Apply parameter-dependent transformation."""
    aux = A - (1.0 - 2.0 * y_deg) * np.fabs(np.floor(0.5 - y_deg) + A)
    ret = np.power(y, B + (C - B) * aux)
    return correct_to_01(ret)


def _transformation_param_deceptive(y, A=0.35, B=0.001, C=0.05):
    """Apply parameter-deceptive transformation."""
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


# ---------------------------------------------------------------------------------------------------------
# REDUCTION
# ---------------------------------------------------------------------------------------------------------


def _reduction_weighted_sum(y, w):
    """Apply weighted sum reduction."""
    return correct_to_01(np.dot(y, w) / w.sum())


def _reduction_weighted_sum_uniform(y):
    """Apply uniform weighted sum reduction (mean)."""
    return correct_to_01(y.mean(axis=1))


def _reduction_non_sep(y, A):
    """Apply non-separable reduction."""
    n, m = y.shape
    val = np.ceil(A / 2.0)

    num = np.zeros(n)
    for j in range(m):
        num += y[:, j]
        for k in range(A - 1):
            num += np.fabs(y[:, j] - y[:, (1 + j + k) % m])

    denom = m * val * (1.0 + 2.0 * A - 2 * val) / A

    return correct_to_01(num / denom)


# ---------------------------------------------------------------------------------------------------------
# SHAPE
# ---------------------------------------------------------------------------------------------------------


def _shape_concave(x, m):
    """Apply concave shape function."""
    M = x.shape[1]
    if m == 1:
        ret = np.prod(np.sin(0.5 * x[:, :M] * np.pi), axis=1)
    elif 1 < m <= M:
        ret = np.prod(np.sin(0.5 * x[:, : M - m + 1] * np.pi), axis=1)
        ret *= np.cos(0.5 * x[:, M - m + 1] * np.pi)
    else:
        ret = np.cos(0.5 * x[:, 0] * np.pi)
    return correct_to_01(ret)


def _shape_convex(x, m):
    """Apply convex shape function."""
    M = x.shape[1]
    if m == 1:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, :M] * np.pi), axis=1)
    elif 1 < m <= M:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, : M - m + 1] * np.pi), axis=1)
        ret *= 1.0 - np.sin(0.5 * x[:, M - m + 1] * np.pi)
    else:
        ret = 1.0 - np.sin(0.5 * x[:, 0] * np.pi)
    return correct_to_01(ret)


def _shape_linear(x, m):
    """Apply linear shape function."""
    M = x.shape[1]
    if m == 1:
        ret = np.prod(x, axis=1)
    elif 1 < m <= M:
        ret = np.prod(x[:, : M - m + 1], axis=1)
        ret *= 1.0 - x[:, M - m + 1]
    else:
        ret = 1.0 - x[:, 0]
    return correct_to_01(ret)


def _shape_mixed(x, A=5.0, alpha=1.0):
    """Apply mixed shape function."""
    aux = 2.0 * A * np.pi
    ret = np.power(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha)
    return correct_to_01(ret)


def _shape_disconnected(x, alpha=1.0, beta=1.0, A=5.0):
    """Apply disconnected shape function."""
    aux = np.cos(A * np.pi * x**beta)
    return correct_to_01(1.0 - x**alpha * aux**2)


# ---------------------------------------------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------------------------------------------


def validate_wfg2_wfg3(l: int) -> None:  # noqa: E741
    """Validate distance parameter for WFG2/WFG3 problems.

    Args:
        l: Distance parameter.

    Raises:
        ValueError: If l is not divisible by 2.
    """
    if not l % 2 == 0:
        raise ValueError(
            "In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2."
        )


def correct_to_01(X, epsilon=1.0e-10):
    """Correct values to valid [0, 1] range with numerical tolerance.

    Args:
        X: Array values to correct.
        epsilon: Numerical tolerance for boundary corrections.

    Returns:
        Corrected array.
    """
    X[np.logical_and(X < 0, X >= 0 - epsilon)] = 0
    X[np.logical_and(X > 1, X <= 1 + epsilon)] = 1
    return X
