import numpy as np
from scipy.linalg import solve_triangular, LinAlgError

from pymoo.algorithms.base.gradient import DerivationBasedAlgorithm, inexact_line_search
from pymoo.algorithms.base.line import LineSearchProblem
from pymoo.algorithms.soo.univariate.wolfe import wolfe_line_search, WolfeSearch
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.solution import Solution


def direction_cholesky(jac, hess):
    # https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf, Page: 510
    adapted = False

    try:
        L = np.linalg.cholesky(hess)

    # hessian is not positive definite, we fix it (for degenerated cases this may lead to "bad" directions)
    except LinAlgError:
        eigvals, _ = np.linalg.eig(hess)
        E = max(0, 1e-5 - eigvals.min()) * np.eye(len(jac))
        try:
            L = np.linalg.cholesky(hess + E)
        # this should happen for very degnerated cases only
        except LinAlgError:
            return None, None, None, False
        adapted = True

    w = solve_triangular(L, - jac, lower=True)
    dir = solve_triangular(L.T, w, lower=False)
    dec = np.linalg.norm(w) ** 2
    return dir, dec, adapted, True


def direction_inv(jac, hess):
    try:
        hess_inv = np.linalg.inv(hess)
        dir = - hess_inv @ jac
        dec = jac.T @ hess_inv @ jac
    except:
        return None, None, False, False
    return dir, dec, False, True




class NewtonMethod(DerivationBasedAlgorithm):

    def __init__(self, damped=True, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.damped = damped
        self.eps = eps

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)
        assert problem.n_obj == 1, "The Newton method can only be used for single-objective problems (n_obj=1)."
        assert problem.n_constr == 0, "The Newton method can only be used for unconstrained problems (n_constr=0)."
        return self

    def _next(self):
        sol = self.opt[0]
        x = sol.get("X")
        jac, hess = self.gradient_and_hessian(sol)

        method = "cholesky"
        func = direction_cholesky if method == "cholesky" else direction_inv
        direction, decrement, adapted, success = func(jac, hess)

        # in case we can not calculate the newton direction fall back to the gradient approach
        if not success:
            direction = -jac
            decrement = np.linalg.norm(direction) ** 2

        if self.damped:
            line = LineSearchProblem(self.problem, sol, direction)
            _next = WolfeSearch().setup(line, evaluator=self.evaluator).run()
            # _next = wolfe_line_search(self.problem, sol, direction, evaluator=self.evaluator)

            # if the newton step was not successful, then try the gradient
            if adapted and (sol.F[0] - _next.F[0]) < 1e-16:
                _next = inexact_line_search(self.problem, sol, -jac, evaluator=self.evaluator)

        else:
            _x = x + direction
            _next = Solution(X=_x)

        if isinstance(_next, Individual):
            _next = Population.create(_next)

        self.pop = Population.merge(self.opt, _next)

        if decrement / 2 <= self.eps:
            self.termination.force_termination = True
