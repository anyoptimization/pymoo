import numpy as np

from pymoo.algorithms.base.local import LocalSearch
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.core.individual import Individual
from pymoo.core.population import Population, pop_from_array_or_individual
from pymoo.core.replacement import is_better
from pymoo.core.termination import Termination
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.util.display import SingleObjectiveDisplay
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
        self.xtol = x_tol
        self.ftol = f_tol
        self.n_max_iter = n_max_iter
        self.n_max_evals = n_max_evals

    def _do_continue(self, algorithm):
        pop, problem = algorithm.pop, algorithm.problem

        if len(pop) != problem.n_var + 1:
            return True

        X, F = pop.get("X", "F")

        ftol = np.abs(F[1:] - F[0]).max() <= self.ftol

        # if the problem has bounds we can normalize the x space to to be more accurate
        if problem.has_bounds():
            xtol = np.abs((X[1:] - X[0]) / (problem.xu - problem.xl)).max() <= self.xtol
        else:
            xtol = np.abs(X[1:] - X[0]).max() <= self.xtol

        # degenerated simplex - get all edges and minimum and maximum length
        D = vectorized_cdist(X, X)
        val = D[np.triu_indices(len(pop), 1)]
        min_e, max_e = val.min(), val.max()

        # either if the maximum length is very small or the ratio is degenerated
        is_degenerated = max_e < 1e-16 or min_e / max_e < 1e-16

        max_iter = algorithm.n_gen > self.n_max_iter
        max_evals = algorithm.evaluator.n_eval > self.n_max_evals

        return not (ftol or xtol or max_iter or max_evals or is_degenerated)


def adaptive_params(problem):
    n = problem.n_var

    alpha = 1
    beta = 1 + 2 / n
    gamma = 0.75 - 1 / (2 * n)
    delta = 1 - 1 / n
    return alpha, beta, gamma, delta


class NelderMead(LocalSearch):

    def __init__(self,
                 func_params=adaptive_params,
                 display=SingleObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        X : np.array or Population
            The initial point where the search should be based on or the complete initial simplex (number of
            dimensions plus 1 points). The population objective can be already evaluated with objective
            space values. If a numpy array is provided it is evaluated by the algorithm.
            By default it is None which means the search starts at a random point.

        func_params : func
            A function that returns the parameters alpha, beta, gamma, delta for the search. By default:

            >>>  def adaptive_params(problem):
            ...     n = problem.n_var
            ...
            ...     alpha = 1
            ...     beta = 1 + 2 / n
            ...     gamma = 0.75 - 1 / (2 * n)
            ...     delta = 1 - 1 / n
            ...     return alpha, beta, gamma, delta

            It can be overwritten if necessary.


        criterion_local_restart : Termination
            Provide a termination object which decides whether a local restart should be performed or not.

        """

        super().__init__(display=display, **kwargs)

        # the function to return the parameter
        self.func_params = func_params

        # the attributes for the simplex operations
        self.alpha, self.beta, self.gamma, self.delta = None, None, None, None

        # the scaling used for the initial simplex
        self.simplex_scaling = None

        # the termination used for nelder and mead if nothing else provided
        self.default_termination = NelderAndMeadTermination()

    def _local_initialize_infill(self):
        self.alpha, self.beta, self.gamma, self.delta = self.func_params(self.problem)

        # the corresponding x values of the provided or best found solution
        x0 = self.x0.X

        # if lower and upper bounds are given take 5% of the range and add
        if self.problem.has_bounds():
            self.simplex_scaling = 0.05 * (self.problem.xu - self.problem.xl)

        # no bounds are given do it based on x0 - MATLAB procedure
        else:
            self.simplex_scaling = 0.05 * self.x0.X
            # some value needs to be added if x0 is zero
            self.simplex_scaling[self.simplex_scaling == 0] = 0.00025

        # initialize the simplex
        simplex = pop_from_array_or_individual(self.initialize_simplex(x0))

        return Population.merge(self.x0, simplex)

    def _local_initialize_advance(self, infills=None, **kwargs):
        # sort the current simplex by fitness
        self.pop = FitnessSurvival().do(self.problem, infills, n_survive=len(infills))

    def _local_advance(self, **kwargs):

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
        reflect = self.evaluator.eval(self.problem, Individual(X=x_reflect), algorithm=self)

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

            # if the expansion is almost equal to reflection (if boundaries were hit) - no evaluation
            if np.allclose(x_expand, x_reflect, atol=1e-16):
                expand = reflect
            else:
                expand = self.evaluator.eval(self.problem, Individual(X=x_expand), algorithm=self)

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
            contract_outside = self.evaluator.eval(self.problem, Individual(X=x_contract_outside), algorithm=self)

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
            contract_inside = self.evaluator.eval(self.problem, Individual(X=x_contract_inside), algorithm=self)

            if is_better(contract_inside, pop[n + 1]):
                pop[n + 1] = contract_inside
            else:
                shrink = True

        # -------------------------------------------------------------------------------------------
        # Shrink (only if necessary)
        # -------------------------------------------------------------------------------------------

        if shrink:
            for i in range(1, len(pop)):
                pop[i].X = pop[0].X + self.delta * (pop[i].X - pop[0].X)
            pop[1:] = self.evaluator.eval(self.problem, pop[1:], algorithm=self)

        self.pop = FitnessSurvival().do(self.problem, pop, n_survive=len(pop))

    def initialize_simplex(self, x0):

        n, xl, xu = self.problem.n_var, self.problem.xl, self.problem.xu

        # repeat the x0 already and add the values
        X = x0[None, :].repeat(n, axis=0)

        for k in range(n):

            # if the problem has bounds do the check
            if self.problem.has_bounds():
                if X[k, k] + self.simplex_scaling[k] < self.problem.xu[k]:
                    X[k, k] = X[k, k] + self.simplex_scaling[k]
                else:
                    X[k, k] = X[k, k] - self.simplex_scaling[k]

            # otherwise just add the scaling
            else:
                X[k, k] = X[k, k] + self.simplex_scaling[k]

        return X


def default_params(*args):
    alpha = 1
    beta = 2.0
    gamma = 0.5
    delta = 0.05
    return alpha, beta, gamma, delta


parse_doc_string(NelderMead.__init__)
