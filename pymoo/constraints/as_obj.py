import numpy as np

import pymoo.gradient.toolbox as anp
from pymoo.core.individual import calc_cv
from pymoo.core.problem import MetaProblem
from pymoo.util.misc import from_dict


class ConstraintsAsObjective(MetaProblem):

    def __init__(self,
                 problem,
                 config=None,
                 append=True):

        super().__init__(problem)
        
        # Store configuration
        self.config = config
        self.append = append

        if append:
            self.n_obj = problem.n_obj + 1
        else:
            self.n_obj = 1

        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def _evaluate(self, X, out, *args, **kwargs):
        # Call the wrapped problem's evaluate method
        wrapped_out = self.__wrapped__.evaluate(X, return_as_dictionary=True, *args, **kwargs)

        # get at the values from the output
        F, G, H = from_dict(wrapped_out, "F", "G", "H")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        # calculate the total constraint violation (here normalization shall be already included)
        CV = calc_cv(G=G, H=H, config=self.config)
        
        # Ensure CV has the right shape
        if isinstance(CV, (int, float)):
            CV = np.full((len(X),), CV)
        elif CV.ndim == 1:
            CV = CV[:, None]

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack([CV, F])
        else:
            out["F"] = CV

    def pareto_front(self, *args, **kwargs):
        pf = self.__wrapped__.pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf
