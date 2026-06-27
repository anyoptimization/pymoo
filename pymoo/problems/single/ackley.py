"""Ackley single-objective benchmark problem."""

from typing import Any

import numpy as np
import pymoo.gradient.toolbox as anp

from pymoo.core.problem import Problem


class Ackley(Problem):
    """Ackley benchmark function for single-objective optimization.

    Args:
        n_var: Number of variables (dimensions).
        a: Parameter a of the Ackley function.
        b: Parameter b of the Ackley function.
        c: Parameter c of the Ackley function.
    """

    def __init__(
        self, n_var: int = 2, a: float = 20, b: float = 1 / 5, c: float = 2 * np.pi
    ) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=-32.768, xu=+32.768, vtype=float)
        self.a = a
        self.b = b
        self.c = c

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        part1 = (
            -1.0
            * self.a
            * anp.exp(
                -1.0 * self.b * anp.sqrt((1.0 / self.n_var) * anp.sum(x * x, axis=1))
            )
        )
        part2 = -1.0 * anp.exp(
            (1.0 / self.n_var) * anp.sum(anp.cos(self.c * x), axis=1)
        )
        out["F"] = part1 + part2 + self.a + anp.exp(1)

    def _calc_pareto_front(self) -> float:
        return 0

    def _calc_pareto_set(self) -> np.ndarray:
        return np.full(self.n_var, 0)
