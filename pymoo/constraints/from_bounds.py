import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.core.meta import Meta
from pymoo.core.problem import MetaProblem


class ConstraintsFromBounds(MetaProblem):
    """
    A problem wrapper that adds boundary constraints.
    Uses Meta for clean delegation and sets the type to appear as a Problem.
    """

    def __init__(self, problem, remove_bonds=False):
        # Don't copy if the problem is already a Meta object to avoid deepcopy issues
        copy = not isinstance(problem, Meta)
        super().__init__(problem, copy=copy)
        
        # Modify constraint count to include boundary constraints
        self.n_ieq_constr += 2 * self.n_var

        if remove_bonds:
            self.xl, self.xu = None, None

    def do(self, X, return_values_of, *args, **kwargs):
        # Call wrapped problem's do method
        out = self.__wrapped__.do(X, return_values_of, *args, **kwargs)

        # Add boundary constraints
        xl, xu = self.bounds()
        _G = anp.column_stack([xl - X, X - xu])
        out["G"] = anp.column_stack([out["G"], _G])

        if "dG" in out:
            _dG = np.zeros((len(X), 2 * self.n_var, self.n_var))
            _dG[:, :self.n_var, :] = -np.eye(self.n_var)
            _dG[:, self.n_var:, :] = np.eye(self.n_var)
            out["dG"] = np.column_stack([out["dG"], _dG])

        return out
