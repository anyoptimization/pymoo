from pymoo.core.problem import calc_constr
from pymoo.problems.meta import MetaProblem
from pymoo.util.misc import from_dict


class ConstraintViolationAsObjective(MetaProblem):

    def __init__(self, problem, eps=1e-6):
        super().__init__(problem)
        self.n_obj = 1
        self.n_constr = 0
        self.eps = eps

    def do(self, x, out, *args, **kwargs):
        super().do(x, out, *args, **kwargs)

        F, G = from_dict(out, "F", "G")

        assert G is not None, "To converge a function's constraint to objective it needs G to be set!"

        out["__F__"] = out["F"]
        out["__G__"] = out["G"]

        out["F"] = calc_constr(G, eps=self.eps, beta=1.0)
        del out["G"]
