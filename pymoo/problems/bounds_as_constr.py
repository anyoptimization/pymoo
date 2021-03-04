import autograd.numpy as anp

from pymoo.problems.meta import MetaProblem


class BoundariesAsConstraints(MetaProblem):

    def __init__(self, problem):
        super().__init__(problem)
        self.n_constr = self.n_constr + 2 * self.n_var

    def do(self, X, out, *args, **kwargs):
        self.problem.do(X, out, *args, **kwargs)

        # get the boundaries for normalization
        xl, xu = self.bounds()

        # add the boundary constraint if enabled
        _G = anp.column_stack([xl-X, X-xu])

        out["G"] = anp.column_stack([out["G"], _G]) if out.get("G") is not None else _G

        if "dG" in out:
            _dG = anp.zeros((len(X), 2 * self.n_var, self.n_var))
            _dG[:, :self.n_var, :] = - anp.eye(self.n_var)
            _dG[:, self.n_var:, :] = anp.eye(self.n_var)
            out["dG"] = anp.column_stack([out["dG"], _dG]) if out.get("dG") is not None else _dG



