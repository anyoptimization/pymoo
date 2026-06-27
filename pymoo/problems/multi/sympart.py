"""SYM-PART test problems."""

from typing import Any

import pymoo.gradient.toolbox as anp
import numpy as np


from pymoo.core.problem import Problem


class SYMPARTRotated(Problem):
    """The SYM-PART test problem proposed in [1].

    Args:
        length: The length of each line (i.e., each Pareto subsets), default is 1.
        v_dist: Vertical distance between the centers of two adjacent lines, default is 10.
        h_dist: Horizontal distance between the centers of two adjacent lines, default is 10.
        angle: The angle to rotate the equivalent Pareto subsets counterclockwise.
            When set to a negative value, Pareto subsets are rotated clockwise.

    References:
        [1] G. Rudolph, B. Naujoks, and M. Preuss, "Capabilities of EMOA to detect and
            preserve equivalent Pareto subsets"
    """

    def __init__(
        self,
        length: float = 1,
        v_dist: float = 10,
        h_dist: float = 10,
        angle: float = np.pi / 4,
    ) -> None:
        self.a = length
        self.b = v_dist
        self.c = h_dist
        self.w = angle

        # Calculate the inverted rotation matrix, store for fitness evaluation
        self.IRM = np.array(
            [[np.cos(self.w), np.sin(self.w)], [-np.sin(self.w), np.cos(self.w)]]
        )

        r = max(self.b, self.c)
        xl = np.full(2, -10 * r)
        xu = np.full(2, 10 * r)

        super().__init__(n_var=2, n_obj=2, vtype=float, xl=xl, xu=xu)

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        if self.w == 0:
            X1 = x[:, 0]
            X2 = x[:, 1]
        else:
            # If rotated, we rotate it back by applying the inverted rotation matrix to x
            Y = anp.array([anp.matmul(self.IRM, xi) for xi in x])
            X1 = Y[:, 0]
            X2 = Y[:, 1]

        a, b, c = self.a, self.b, self.c
        t1_hat = anp.sign(X1) * anp.ceil((anp.abs(X1) - a - c / 2) / (2 * a + c))
        t2_hat = anp.sign(X2) * anp.ceil((anp.abs(X2) - b / 2) / b)
        one = anp.ones(len(x))
        t1 = anp.sign(t1_hat) * anp.min(anp.vstack((anp.abs(t1_hat), one)), axis=0)
        t2 = anp.sign(t2_hat) * anp.min(anp.vstack((anp.abs(t2_hat), one)), axis=0)

        p1 = X1 - t1 * c
        p2 = X2 - t2 * b

        f1 = (p1 + a) ** 2 + p2**2
        f2 = (p1 - a) ** 2 + p2**2
        out["F"] = anp.vstack((f1, f2)).T

    def _calc_pareto_set(self, n_pareto_points: int = 500) -> np.ndarray:
        # The SYM-PART test problem has 9 equivalent Pareto subsets.
        h = int(n_pareto_points / 9)
        PS = np.zeros((h * 9, self.n_var))
        cnt = 0
        for row in [-1, 0, 1]:
            for col in [1, 0, -1]:
                X1 = np.linspace(row * self.c - self.a, row * self.c + self.a, h)
                X2 = np.tile(col * self.b, h)
                PS[cnt * h : cnt * h + h, :] = np.vstack((X1, X2)).T
                cnt = cnt + 1
        if self.w != 0:
            # If rotated, we apply the rotation matrix to PS
            # Calculate the rotation matrix
            RM = np.array(
                [[np.cos(self.w), -np.sin(self.w)], [np.sin(self.w), np.cos(self.w)]]
            )
            PS = np.array([np.matmul(RM, x) for x in PS])
        return PS

    def _calc_pareto_front(self, n_pareto_points: int = 500) -> np.ndarray:
        PS = self.pareto_set(n_pareto_points)
        return self.evaluate(PS, return_values_of=["F"])


class SYMPART(SYMPARTRotated):
    """SYM-PART test problem (non-rotated)."""

    def __init__(
        self, length: float = 1, v_dist: float = 10, h_dist: float = 10
    ) -> None:
        super().__init__(length, v_dist, h_dist, 0)
