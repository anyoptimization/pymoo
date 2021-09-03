from pymoo.algorithms.base.line import LineSearchProblem
from pymoo.algorithms.base.local import LocalSearch
from pymoo.algorithms.soo.univariate.backtracking import BacktrackingLineSearch
from pymoo.algorithms.soo.univariate.golden import GoldenSectionSearch
from pymoo.optimize import minimize

from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination


class DerivationBasedDisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("alpha", "-" if algorithm.alpha is None else algorithm.alpha)


class DerivationBasedAlgorithm(LocalSearch):

    def __init__(self, strict_bounds=True, requires_hessian=False, display=DerivationBasedDisplay(), **kwargs):
        super().__init__(display=display, **kwargs)

        # whether the bounds are treated to be strict
        self.strict_bounds = strict_bounds

        # whether the implementation requires the calculation of the hessian matrix
        self.requires_hessian = requires_hessian

        # gradient of the solution in the current iteration
        self.dF = None

        # the direction that has been used for the line search
        self.dir = None

        # the resulting scalar from the line search
        self.alpha = 1.0

        # set the default termination
        self.default_termination = SingleObjectiveSpaceToleranceTermination(n_last=3, tol=1e-10)

    def _setup(self, problem, **kwargs):
        assert problem.n_obj == 1, "The Newton method can only be used for single-objective problems (n_obj=1)."
        assert problem.n_constr == 0, "The Newton method can only be used for unconstrained problems (n_constr=0)."



def exact_line_search(problem, sol, direction, algorithm=GoldenSectionSearch(), n_iter=10, evaluator=None):
    problem = LineSearchProblem(problem, sol, direction)
    res = minimize(problem, algorithm, ("n_iter", n_iter), evaluator=evaluator)
    sol = res.opt[0]
    sol.set("X", sol.get("__X__"))
    return sol


def inexact_line_search(problem, sol, direction, evaluator=None):
    algorithm = BacktrackingLineSearch()
    algorithm = algorithm.setup(problem, point=sol, direction=direction)._initialize()
    res = minimize(problem, algorithm, evaluator=evaluator)
    return res.opt[0]
