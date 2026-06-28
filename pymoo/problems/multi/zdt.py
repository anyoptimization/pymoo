"""ZDT benchmark test problems."""

from typing import Any

import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.core.problem import Problem
from pymoo.util.normalization import normalize


class ZDT(Problem):
    """Base class for ZDT test problems."""

    def __init__(self, n_var: int = 30, **kwargs: Any) -> None:
        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=1, vtype=float, **kwargs)


class ZDT1(ZDT):
    """ZDT1 test problem."""

    def _calc_pareto_front(self, n_pareto_points: int = 100) -> np.ndarray:
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * anp.sum(x[:, 1:], axis=1)
        f2 = g * (1 - anp.power((f1 / g), 0.5))

        out["F"] = anp.column_stack([f1, f2])


class ZDT2(ZDT):
    """ZDT2 test problem."""

    def _calc_pareto_front(self, n_pareto_points: int = 100) -> np.ndarray:
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.power(x, 2)]).T

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        f1 = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - anp.power((f1 * 1.0 / g), 2))

        out["F"] = anp.column_stack([f1, f2])


class ZDT3(ZDT):
    """ZDT3 test problem with multiple disconnected Pareto fronts."""

    def _calc_pareto_front(self, n_points: int = 100, flatten: bool = True) -> np.ndarray:
        regions = [
            [0, 0.0830015349],
            [0.182228780, 0.2577623634],
            [0.4093136748, 0.4538821041],
            [0.6183967944, 0.6525117038],
            [0.8233317983, 0.8518328654],
        ]

        pf_list: list[np.ndarray] = []

        for r in regions:
            x1 = np.linspace(r[0], r[1], int(n_points / len(regions)))
            x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
            pf_list.append(np.array([x1, x2]).T)

        if not flatten:
            pf = np.concatenate([pf_item[None, ...] for pf_item in pf_list])
        else:
            pf = np.vstack(pf_list)

        return pf

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        f1 = x[:, 0]
        c = anp.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - anp.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * anp.sin(10 * anp.pi * f1))

        out["F"] = anp.column_stack([f1, f2])


class ZDT4(ZDT):
    """ZDT4 test problem with multimodal landscape."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var)
        self.xl = -5 * np.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 5 * np.ones(self.n_var)
        self.xu[0] = 1.0
        self.func = self._evaluate

    def _calc_pareto_front(self, n_pareto_points: int = 100) -> np.ndarray:
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        f1 = x[:, 0]
        g = 1.0
        g += 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[:, i] * x[:, i] - 10.0 * anp.cos(4.0 * anp.pi * x[:, i])
        h = 1.0 - anp.sqrt(f1 / g)
        f2 = g * h

        out["F"] = anp.column_stack([f1, f2])


class ZDT5(ZDT):
    """ZDT5 test problem with binary variables."""

    def __init__(self, m: int = 11, n: int = 5, normalize: bool = True, **kwargs: Any) -> None:
        self.m = m
        self.n = n
        self.normalize = normalize
        super().__init__(n_var=(30 + n * (m - 1)), **kwargs)

    def _calc_pareto_front(self, n_pareto_points: int = 100) -> np.ndarray:
        x = 1 + np.linspace(0, 1, n_pareto_points) * 30
        pf = np.column_stack([x, (self.m - 1) / x])
        if self.normalize:
            pf = normalize(pf)
        return pf

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        x = x.astype(float)

        _x = [x[:, :30]]
        for i in range(self.m - 1):
            _x.append(x[:, 30 + i * self.n : 30 + (i + 1) * self.n])

        u = anp.column_stack([x_i.sum(axis=1) for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[:, 1:].sum(axis=1)

        f1 = 1 + u[:, 0]
        f2 = g * (1 / f1)

        if self.normalize:
            f1 = normalize(f1, 1, 31)
            f2 = normalize(f2, (self.m - 1) * 1 / 31, (self.m - 1))

        out["F"] = anp.column_stack([f1, f2])


class ZDT6(ZDT):
    """ZDT6 test problem with non-uniform spacing."""

    def __init__(self, n_var: int = 10, **kwargs: Any) -> None:
        super().__init__(n_var=n_var, **kwargs)

    def _calc_pareto_front(self, n_pareto_points: int = 100) -> np.ndarray:
        x = np.linspace(0.2807753191, 1, n_pareto_points)
        return np.array([x, 1 - np.power(x, 2)]).T

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        f1 = 1 - anp.exp(-4 * x[:, 0]) * anp.power(anp.sin(6 * anp.pi * x[:, 0]), 6)
        g = 1 + 9.0 * anp.power(anp.sum(x[:, 1:], axis=1) / (self.n_var - 1.0), 0.25)
        f2 = g * (1 - anp.power(f1 / g, 2))

        out["F"] = anp.column_stack([f1, f2])
