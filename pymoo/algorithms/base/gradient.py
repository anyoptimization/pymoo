from pymoo.algorithms.base.line import LineSearchProblem
from pymoo.algorithms.base.local import LocalSearch
from pymoo.algorithms.soo.univariate.backtracking import BacktrackingLineSearch
from pymoo.algorithms.soo.univariate.golden import GoldenSectionSearch
from pymoo.optimize import minimize
from pymoo.util.differentiation.numerical import NumericalDifferentiation
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination


class GradientBasedAlgorithm(LocalSearch):

    def __init__(self, requires_hessian=False, **kwargs):
        super().__init__(display=SingleObjectiveDisplay(), **kwargs)
        self.requires_hessian = requires_hessian
        self.termination = SingleObjectiveSpaceToleranceTermination(n_last=3, tol=1e-12)

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)
        assert problem.n_obj == 1, "The Newton method can only be used for single-objective problems (n_obj=1)."
        assert problem.n_constr == 0, "The Newton method can only be used for unconstrained problems (n_constr=0)."

    def gradient(self, sol):
        dF = sol.get("dF")
        numdiff = NumericalDifferentiation()
        if dF is None:
            dF = numdiff.do(self.problem, sol, evaluator=self.evaluator, return_values_of=["dF"])
        return dF[0]

    def gradient_and_hessian(self, sol):
        dF, ddF = sol.get("dF", "ddF")
        numdiff = NumericalDifferentiation()
        if dF is None or ddF is None:
            dF, ddF = numdiff.do(self.problem, sol, hessian=True, evaluator=self.evaluator,
                                 return_values_of=["dF", "ddF"])
        return dF[0], ddF[0]


def exact_line_search(problem, sol, direction, algorithm=GoldenSectionSearch(), n_iter=10, evaluator=None):
    problem = LineSearchProblem(problem, sol, direction)
    res = minimize(problem, algorithm, ("n_iter", n_iter), evaluator=evaluator)
    sol = res.opt[0]
    sol.set("X", sol.get("__X__"))
    return sol

def inexact_line_search(problem, sol, direction, evaluator=None):
    algorithm = BacktrackingLineSearch()
    algorithm = algorithm.setup(problem, point=sol, direction=direction).initialize()
    res = minimize(problem, algorithm, evaluator=evaluator)
    return res.opt[0]
