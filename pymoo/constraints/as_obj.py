import autograd.numpy as anp
import numpy as np

from pymoo.core.individual import calc_cv
from pymoo.core.meta import Meta
from pymoo.core.problem import Problem, defaults_of_out
from pymoo.util.misc import from_dict


class ConstraintsAsObjective(Meta, Problem):

    def __init__(self,
                 problem,
                 append=True):

        super().__init__(problem)
        self.append = append

        if append:
            self.n_obj = problem.n_obj + 1
        else:
            self.n_obj = 1

        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def do(self, X, return_values_of, *args, **kwargs):

        out = self.__object__.do(X, return_values_of, *args, **kwargs)

        # get at the values from the output
        F, G, H = from_dict(out, "F", "G", "H")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        # calculate the total constraint violation (here normalization shall be already included)
        CV = np.array([calc_cv(g, h) for g, h in zip(G, H)])

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack([CV, F])
        else:
            out["F"] = CV

        DEFAULTS = defaults_of_out(self, len(X))
        out["G"] = DEFAULTS["G"]()
        out["H"] = DEFAULTS["H"]()

        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf
