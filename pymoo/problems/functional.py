
import numpy as np

from pymoo.core.problem import ElementwiseProblem


def func_return_none(*args, **kwargs):
    return None


class FunctionalProblem(ElementwiseProblem):

    def __init__(self,
                 n_var,
                 objs,
                 constr_ieq=[],
                 constr_eq=[],
                 func_pf=func_return_none,
                 func_ps=func_return_none,
                 **kwargs):

        # if only a single callable is provided (for single-objective problems) convert it to a list
        if callable(objs):
            objs = [objs]

        self.objs = objs
        self.constr_ieq = constr_ieq
        self.constr_eq = constr_eq
        self.func_pf = func_pf
        self.func_ps = func_ps

        super().__init__(n_var=n_var,
                         n_obj=len(self.objs),
                         n_ieq_constr=len(constr_ieq),
                         n_eq_constr=len(constr_eq),
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([obj(x) for obj in self.objs])
        out["G"] = np.array([constr(x) for constr in self.constr_ieq])
        out["H"] = np.array([constr(x) for constr in self.constr_eq])

    def _calc_pareto_front(self, *args, **kwargs):
        return self.func_pf(*args, **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        return self.func_ps(*args, **kwargs)

