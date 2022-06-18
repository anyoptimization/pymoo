import autograd.numpy as anp

from pymoo.core.meta import Meta
from pymoo.core.problem import Problem


class ConstraintsFromBounds(Meta, Problem):

    def __init__(self, problem, remove_bonds=False):
        super().__init__(problem)
        self.n_ieq_constr += 2 * self.n_var

        if remove_bonds:
            self.xl, self.xu = None, None

    def do(self, X, return_values_of, *args, **kwargs):

        out = self.__object__.do(X, return_values_of, *args, **kwargs)

        # get the boundaries for normalization
        xl, xu = self.bounds()

        # add the boundary constraint if enabled
        _G = anp.column_stack([xl - X, X - xu])

        out["G"] = anp.column_stack([out["G"], _G])

        if "dG" in out:
            _dG = anp.zeros((len(X), 2 * self.n_var, self.n_var))
            _dG[:, :self.n_var, :] = - anp.eye(self.n_var)
            _dG[:, self.n_var:, :] = anp.eye(self.n_var)
            out["dG"] = anp.column_stack([out["dG"], _dG]) if out.get("dG") is not None else _dG

        return out
