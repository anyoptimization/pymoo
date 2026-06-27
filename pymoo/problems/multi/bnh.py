"""BNH constrained multi-objective test problem."""

import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.problem import Problem


class BNH(Problem):
    """Binh and Korn (BNH) constrained multi-objective test problem.

    A 2-variable, 2-objective optimization problem with 2 inequality constraints.
    """

    def __init__(self):
        """Initialize BNH problem with 2 variables, 2 objectives, and 2 constraints."""
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=2, vtype=float)
        self.xl = np.zeros(self.n_var)
        self.xu = np.array([5.0, 3.0])

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate BNH problem objectives and constraints."""
        f1 = 4 * x[:, 0] ** 2 + 4 * x[:, 1] ** 2
        f2 = (x[:, 0] - 5) ** 2 + (x[:, 1] - 5) ** 2
        g1 = (1 / 25) * ((x[:, 0] - 5) ** 2 + x[:, 1] ** 2 - 25)
        g2 = -1 / 7.7 * ((x[:, 0] - 8) ** 2 + (x[:, 1] + 3) ** 2 - 7.7)

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_front(self, n_points=100):
        """Calculate Pareto front approximation."""
        x1 = np.linspace(0, 5, n_points)
        x2 = np.linspace(0, 5, n_points)
        x2[x1 >= 3] = 3

        X = np.column_stack([x1, x2])
        return self.evaluate(X, return_values_of=["F"])
