import pymoo.gradient.toolbox as anp
import numpy as np

from pymoo.core.individual import calc_cv
from pymoo.core.problem import Problem
from pymoo.util.misc import from_dict


class ConstraintsAsObjective(Problem):

    def __init__(self,
                 problem,
                 config=None,
                 append=True):

        super().__init__()
        
        # Store the wrapped problem
        self.problem = problem
        self.config = config
        self.append = append
        
        # Copy relevant attributes from the wrapped problem
        self.n_var = problem.n_var
        self.xl = getattr(problem, 'xl', None)
        self.xu = getattr(problem, 'xu', None)
        
        # Copy other important attributes
        for attr in ['elementwise', 'parallelization', 'replace_nan_values_by']:
            if hasattr(problem, attr):
                setattr(self, attr, getattr(problem, attr))

        if append:
            self.n_obj = problem.n_obj + 1
        else:
            self.n_obj = 1

        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.problem.do(X, return_values_of, *args, **kwargs)

        # get at the values from the output
        F, G, H = from_dict(out, "F", "G", "H")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        # calculate the total constraint violation (here normalization shall be already included)
        CV = calc_cv(G=G, H=H, config=self.config)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack([CV, F])
        else:
            out["F"] = CV

        del out["G"]
        del out["H"]

        return out

    def pareto_front(self, *args, **kwargs):
        pf = self.problem.pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf
