import autograd.numpy as anp
import numpy as np

from pymoo.model.problem import Problem


class WFG(Problem):

    def __init__(self, k, l, n_obj, **kwargs):
        super().__init__(n_var=k+l, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=anp.double, **kwargs)


class WFG1(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))


class WFG2(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))


class WFG3(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))


class WFG4(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))


class WFG5(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))


class WFG6(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))


class WFG7(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))


class WFG8(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))


class WFG9(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.random.random((len(x), self.n_obj))






