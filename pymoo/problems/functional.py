
from pymoo.model.problem import Problem


def func_return_none(*args, **kwargs):
    return None


class FunctionalProblem(Problem):

    def __init__(self,
                 n_var,
                 objs,
                 constr_ieq=[],
                 constr_eq=[],
                 constr_eq_eps=1e-6,
                 func_pf=func_return_none,
                 func_ps=func_return_none,
                 **kwargs):
        if callable(objs):
            objs = [objs]
        self.objs = objs
        self.constr_ieq = constr_ieq
        self.constr_eq = constr_eq
        self.constr_eq_eps = constr_eq_eps
        self.func_pf = func_pf
        self.func_ps = func_ps

        n_constr = len(constr_ieq) + len(constr_eq)

        super().__init__(n_var,
                         n_obj=len(self.objs),
                         n_constr=n_constr,
                         elementwise_evaluation=True,
                         **kwargs)
