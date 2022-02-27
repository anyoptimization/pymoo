from pymoo.problems.meta import MetaProblem
from pymoo.util.misc import at_least_2d_array
import autograd.numpy as anp


class ConstraintsAsObjective(MetaProblem):

    def __init__(self,
                 problem,
                 tcv,
                 append=True):

        super().__init__(problem)
        self.append = append

        if append:
            self.n_obj = problem.n_obj + 1
        else:
            self.n_obj = 1

        self.n_ieq_constr = 0
        self.n_eq_constr = 0
        self.tcv = tcv

    def do(self, x, out, *args, **kwargs):
        super().do(x, out, *args, **kwargs)

        F, G, H = at_least_2d_array(out["F"]), at_least_2d_array(out["G"]), at_least_2d_array(out["H"])

        # store a backup of the values in out
        out["__G__"], out["__H__"] = G, H

        # calculate the total constraint violation (here normalization shall be already included)
        cv = self.tcv.calc(G, H)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack([F, cv])
        else:
            out["F"] = cv

        # erase the values from the output - it is unconstrained now
        out.pop("G", None), out.pop("H", None)
