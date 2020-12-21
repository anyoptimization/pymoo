import numpy as np

from pymoo.model.problem import MetaProblem


class BoundariesAsConstraints(MetaProblem):

    def __init__(self, problem):
        super().__init__(problem)
        self.n_constr = self.n_constr + 2 * self.n_var

    def do(self, X, out, *args, **kwargs):
        super().do(X, out, *args, **kwargs)

        # get the boundaries for normalization
        xl, xu = self.bounds()

        # add the boundary constraint if enabled
        _G = np.zeros((len(X), 2 * self.n_var))
        _G[:, :self.n_var] = (xl - X)
        _G[:, self.n_var:] = (X - xu)

        out["G"] = np.column_stack([out["G"], _G]) if out.get("G") is not None else _G

        if "dG" in out:
            _dG = np.zeros((len(X), 2 * self.n_var, self.n_var))
            _dG[:, :self.n_var, :] = - np.eye(self.n_var)
            _dG[:, self.n_var:, :] = np.eye(self.n_var)
            out["dG"] = np.column_stack([out["dG"], _dG]) if out.get("dG") is not None else _dG



