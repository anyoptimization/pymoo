from pymoo.problems.meta import MetaProblem
from pymoo.util.misc import at_least_2d_array


class ConstraintsAsPenalty(MetaProblem):

    def __init__(self,
                 problem,
                 tcv,
                 penalty: float = 0.1,
                 f_ideal=None,
                 f_scale=None
                 ):

        super().__init__(problem)

        # the amount of penalty to add for this type
        self.penalty = penalty

        # the cv calculator to obtain the constraint violation from G and H
        self.tcv = tcv

        # normalization parameters for the objective value(s)
        self.f_ideal = f_ideal
        self.f_scale = f_scale

        # set ieq and eq to zero (because it became now a penalty)
        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def do(self, x, out, *args, **kwargs):
        self.problem.do(x, out, *args, **kwargs)

        if self.problem.has_constraints():

            # get at the values from the output
            F, G, H = at_least_2d_array(out["F"]), at_least_2d_array(out["G"]), at_least_2d_array(out["H"])

            # store a backup of the values in out
            out["__F__"], out["__G__"], out["__H__"] = F, G, H

            # normalize the objective space values if more information is provided
            if self.f_ideal is not None:
                F = F - self.f_ideal
            if self.f_scale is not None:
                F = F / self.f_scale

            # calculate the total constraint violation (here normalization shall be already included)
            cv = self.tcv.calc(G, H)

            # set the penalized objective values
            out["F"] = F + self.penalty * cv[:, None]

            # erase the values from the output - it is unconstrained now
            out.pop("G", None), out.pop("H", None)
