import numpy as np
from scipy.linalg import solve_triangular, LinAlgError

from pymoo.algorithms.base.gradient import GradientBasedAlgorithm
from pymoo.model.population import Population
from pymoo.model.solution import Solution
from pymoo.optimize import minimize
from pymoo.problems.single import Himmelblau


class NewtonMethod(GradientBasedAlgorithm):

    def __init__(self, damped=True, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.damped = damped
        self.eps = eps

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)
        assert problem.n_obj == 1, "The Newton method can only be used for single-objective problems (n_obj=1)."
        assert problem.n_constr == 0, "The Newton method can only be used for unconstrained problems (n_constr=0)."

    def _next(self):
        sol = self.opt[0]
        x = sol.get("X")
        dF, ddF = self.gradient_and_hessian(sol)

        # ddF_inv = np.linalg.inv(ddF)
        # direction = - ddF_inv @ dF
        # decrement = dF.T @ ddF_inv @ dF
        # https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf, Page: 510

        try:
            L = np.linalg.cholesky(ddF)
            is_pos_def = True

        # hessian is not positive definite, we fix it (for degenerated cases this may lead to "bad" directions)
        except LinAlgError:
            eigvals, _ = np.linalg.eig(ddF)
            E = max(0, 1e-8 - eigvals.min()) * np.eye(len(x))
            L = np.linalg.cholesky(ddF + E)
            is_pos_def = False

        w = solve_triangular(L, - dF, lower=True)
        direction = solve_triangular(L.T, w, lower=False)
        decrement = np.linalg.norm(w) ** 2

        if self.damped:
            _next = self.inexact_line_search(sol, direction)

            # if the newton step was not successful, then try the gradient
            if not is_pos_def and (sol.F[0] - _next.F[0]) < 1e-16:
                _next = self.inexact_line_search(sol, -dF)

        else:
            _x = x + direction
            _next = Solution(X=_x)

        self.pop = Population.merge(self.opt, _next)

        if decrement / 2 <= self.eps:
            self.termination.force_termination = True


problem = Himmelblau()

algorithm = NewtonMethod()

res = minimize(problem, algorithm, verbose=True, seed=1)
