
import autograd.numpy as np

from pymoo.core.problem import ElementwiseProblem


def func_return_none(*args, **kwargs):
    return None


class FunctionalProblem(ElementwiseProblem):

    def __init__(self,
                 n_var,
                 objs,
                 constr_ieq=[],
                 constr_eq=[],
                 constr_eq_eps=1e-6,
                 func_pf=func_return_none,
                 func_ps=func_return_none,
                 **kwargs):

        # if only a single callable is provided (for single-objective problems) convert it to a list
        if callable(objs):
            objs = [objs]

        self.objs = objs
        self.constr_ieq = constr_ieq
        self.constr_eq = constr_eq
        self.constr_eq_eps = constr_eq_eps
        self.func_pf = func_pf
        self.func_ps = func_ps

        n_constr = len(constr_ieq) + len(constr_eq)

        super().__init__(n_var=n_var,
                         n_obj=len(self.objs),
                         n_constr=n_constr,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        # calculate violation from the inequality constraints
        ieq = np.array([constr(x) for constr in self.constr_ieq])
        ieq[ieq < 0] = 0

        # calculate violation from the quality constraints
        eq = np.array([constr(x) for constr in self.constr_eq])
        eq = np.abs(eq)
        eq = eq - self.constr_eq_eps

        # calculate the objective function
        f = np.array([obj(x) for obj in self.objs])

        out["F"] = f
        out["G"] = np.concatenate([ieq, eq])

    def _calc_pareto_front(self, *args, **kwargs):
        return self.func_pf(*args, **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        return self.func_ps(*args, **kwargs)

