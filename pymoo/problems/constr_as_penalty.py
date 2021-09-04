from pymoo.core.problem import Problem
from pymoo.problems.meta import MetaProblem
from pymoo.util.misc import at_least_2d_array


class ConstraintsAsPenalty(MetaProblem):

    def __init__(self, problem, penalty=1e6):
        super().__init__(problem)
        self.penalty = penalty

        # set the constraints to be zero, because they are now added to the objective
        self.n_constr = 0

    def do(self, x, out, *args, **kwargs):
        self.problem.do(x, out, *args, **kwargs)

        if self.problem.has_constraints():

            F, G = at_least_2d_array(out["F"]), at_least_2d_array(out["G"])
            CV = Problem.calc_constraint_violation(G)

            out["__F__"] = F
            out["__G__"] = G
            out["__CV__"] = CV

            out["F"] = F + self.penalty * CV
            out["G"] = None

