import numpy as np

from pymoo.algorithms.base.local import LocalSearch
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.population import pop_from_array_or_individual
from pymoo.core.replacement import is_better
from pymoo.core.termination import Termination
from pymoo.docs import parse_doc_string
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.misc import vectorized_cdist
from pymoo.util.vectors import max_alpha


# =========================================================================================================
# Implementation
# =========================================================================================================


class NelderAndMeadTermination(Termination):

    def __init__(self,
                 x_tol=1e-6,
                 f_tol=1e-6,
                 n_max_iter=1e6,
                 n_max_evals=1e6):

        super().__init__()
        self.x_tol = x_tol
        self.f_tol = f_tol
        self.n_max_iter = n_max_iter
        self.n_max_evals = n_max_evals

    def _update(self, algorithm):
        pop, problem = algorithm.pop, algorithm.problem

        if len(pop) <= 1:
            return 0.0

        X, F = pop.get("X", "F")

        f_delta = np.abs(F[1:] - F[0]).max()
        f_tol = 1 / (1 + (f_delta - self.f_tol))

        # if the problem has bounds we can normalize the x space to to be more accurate
        if problem.has_bounds():
            x_delta = np.abs((X[1:] - X[0]) / (problem.xu - problem.xl)).max()
        else:
            x_delta = np.abs(X[1:] - X[0]).max()

        x_tol = 1 / (1 + (x_delta - self.x_tol))

        # degenerated simplex - get all edges and minimum and maximum length
        D = vectorized_cdist(X, X)
        val = D[np.triu_indices(len(pop), 1)]
        min_e, max_e = val.min(), val.max()

        # either if the maximum length is very small or the ratio is degenerated
        is_degenerated = int(max_e < 1e-16 or min_e / max_e < 1e-16)

        max_iter = algorithm.n_iter / self.n_max_iter
        max_evals = algorithm.evaluator.n_eval / self.n_max_evals

        return max(f_tol, x_tol, max_iter, max_evals, is_degenerated)


def adaptive_params(problem):
    n = problem.n_var
    alpha = 1
    beta = 1 + 2 / n
    gamma = 0.75 - 1 / (2 * n)
    delta = 1 - 1 / n
    return alpha, beta, gamma, delta


def default_params(_):
    alpha = 1
    beta = 2.0
    gamma = 0.5
    delta = 0.05
    return alpha, beta, gamma, delta


def initialize_simplex(problem, x0, scale=0.05):
    n = len(x0)

    if problem.has_bounds():
        delta = scale * (problem.xu - problem.xl)
    else:
        delta = scale * x0
        delta[delta == 0] = 0.00025

    # repeat the x0 already and add the values
    X = x0[None, :].repeat(n, axis=0)

    for k in range(n):

        # if the problem has bounds do the check
        if problem.has_bounds():
            if X[k, k] + delta[k] < problem.xu[k]:
                X[k, k] = X[k, k] + delta[k]
            else:
                X[k, k] = X[k, k] - delta[k]

        # otherwise just add the init_simplex_scale
        else:
            X[k, k] = X[k, k] + delta[k]

    return X


class NelderMead(LocalSearch):

    def __init__(self,
                 init_simplex_scale=0.05,
                 func_params=adaptive_params,
                 output=SingleObjectiveOutput(),
                 **kwargs):

        super().__init__(output=output, **kwargs)

        # the function to return the parameter
        self.func_params = func_params

        # the attributes for the simplex operations
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None

        # whether the simplex has been initialized or not
        self.is_simplex_initialized = False

        # the initial simplex scale used
        self.init_simplex_scale = init_simplex_scale

        # the termination used for nelder and mead if nothing else provided
        self.termination = NelderAndMeadTermination()

    def _setup(self, problem, **kwargs):
        self.alpha, self.beta, self.gamma, self.delta = self.func_params(self.problem)

    def _initialize_simplex(self):
        simplex = pop_from_array_or_individual(initialize_simplex(self.problem, self.x0.X, scale=0.05))
        return Population.merge(self.x0, simplex)

    def _next(self):
        if not self.is_simplex_initialized:
            self.pop = yield self._initialize_simplex()
            self.is_simplex_initialized = True
        else:
            yield from self._step()

    def _step(self):

        # number of variables increased by one - matches equations in the paper
        xl, xu = self.problem.bounds()
        pop, n = self.pop, self.problem.n_var - 1

        # calculate the centroid
        centroid = pop[:n + 1].get("X").mean(axis=0)

        # -------------------------------------------------------------------------------------------
        # REFLECT
        # -------------------------------------------------------------------------------------------

        # check the maximum alpha until the bounds are hit
        alphaU = max_alpha(centroid, (centroid - pop[n + 1].X), xl, xu)

        # reflect the point, consider factor if bounds are there, make sure in bounds (floating point) evaluate
        x_reflect = centroid + min(self.alpha, alphaU) * (centroid - pop[n + 1].X)
        x_reflect = set_to_bounds_if_outside_by_problem(self.problem, x_reflect)
        reflect = yield Individual(X=x_reflect)

        # whether a shrink is necessary or not - decided during this step
        shrink = False

        better_than_current_best = is_better(reflect, pop[0])
        better_than_second_worst = is_better(reflect, pop[n])
        better_than_worst = is_better(reflect, pop[n + 1])

        # if better than the current best - check for expansion
        if better_than_current_best:

            # -------------------------------------------------------------------------------------------
            # EXPAND
            # -------------------------------------------------------------------------------------------

            # the maximum expansion until the bounds are hit
            betaU = max_alpha(centroid, (x_reflect - centroid), xl, xu)

            # expand using the factor, consider bounds, make sure in case of floating point issues
            x_expand = centroid + min(self.beta, betaU) * (x_reflect - centroid)
            x_expand = set_to_bounds_if_outside_by_problem(self.problem, x_expand)
            expand = yield Individual(X=x_expand)

            # if the expansion further improved take it - otherwise use expansion
            if is_better(expand, reflect):
                pop[n + 1] = expand
            else:
                pop[n + 1] = reflect

        # if the new point is not better than the best, but better than second worst - just keep it
        elif not better_than_current_best and better_than_second_worst:
            pop[n + 1] = reflect

        # if not worse than the worst - outside contraction
        elif not better_than_second_worst and better_than_worst:

            # -------------------------------------------------------------------------------------------
            # Outside Contraction
            # -------------------------------------------------------------------------------------------

            x_contract_outside = centroid + self.gamma * (x_reflect - centroid)
            contract_outside = yield Individual(X=x_contract_outside)

            if is_better(contract_outside, reflect):
                pop[n + 1] = contract_outside
            else:
                shrink = True

        # if the reflection was worse than the worst - inside contraction
        else:

            # -------------------------------------------------------------------------------------------
            # Inside Contraction
            # -------------------------------------------------------------------------------------------

            x_contract_inside = centroid - self.gamma * (x_reflect - centroid)
            contract_inside = yield Individual(X=x_contract_inside)

            if is_better(contract_inside, pop[n + 1]):
                pop[n + 1] = contract_inside
            else:
                shrink = True

        # -------------------------------------------------------------------------------------------
        # Shrink (only if necessary)
        # -------------------------------------------------------------------------------------------

        if shrink:
            x_best, x_others = pop[0].X, pop[1:].get("X")
            x_shrink = x_best + self.delta * (x_others - x_best)
            pop[1:] = yield Population.new(X=x_shrink)

        self.pop = FitnessSurvival().do(self.problem, pop, n_survive=len(pop))


parse_doc_string(NelderMead.__init__)
