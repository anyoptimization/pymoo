import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.model.termination import Termination
from pymoo.operators.repair.out_of_bounds_repair import repair_out_of_bounds
from pymoo.util.display import disp_single_objective
from pymoo.util.misc import vectorized_cdist, evaluate_if_not_done_yet
from pymoo.util.normalization import denormalize


# =========================================================================================================
# Implementation
# =========================================================================================================


def default_params(*args):
    alpha = 1
    beta = 2.0
    gamma = 0.5
    delta = 0.05
    return alpha, beta, gamma, delta


def adaptive_params(problem):
    n = problem.n_var

    alpha = 1
    beta = 1 + 2 / n
    gamma = 0.75 - 1 / (2 * n)
    delta = 1 - 1 / n
    return alpha, beta, gamma, delta


class NelderAndMeadTermination(Termination):

    def __init__(self,
                 xtol=1e-6,
                 ftol=1e-6,
                 n_max_iter=1e6,
                 n_max_evals=1e6):

        super().__init__()
        self.xtol = xtol
        self.ftol = ftol
        self.n_max_iter = n_max_iter
        self.n_max_evals = n_max_evals

    def _do_continue(self, algorithm):
        pop, problem = algorithm.pop, algorithm.problem

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

    def do_restart(self, algorithm):
        return self.has_finished(algorithm)


def max_expansion_factor(point, direction, xl, xu):
    bounds = []

    if xl is not None:
        bounds.append(xl)

    if xu is not None:
        bounds.append(xu)

    if len(bounds) == 0:
        return np.inf

    # the bounds in one array
    bounds = np.column_stack(bounds)

    # if the direction is too small we can not divide by 0 - nan will make it being ignored
    _direction = direction.copy()
    _direction[_direction == 0] = np.nan

    # calculate the max factor to be not out of bounds
    val = (bounds - point[:, None]) / _direction[:, None]

    # remove nan and less than 0 values
    val = val[np.logical_not(np.isnan(val))]
    val = val[val >= 0]

    # if no value there - no bound exist
    if len(val) == 0:
        return np.inf
    # otherwise return the minimum of values considered
    else:
        return val.min()


class NelderMead(Algorithm):

    def __init__(self,
                 x0=None,
                 func_params=adaptive_params,
                 termination=NelderAndMeadTermination(xtol=1e-6, ftol=1e-6, n_max_iter=1e6, n_max_evals=1e6),
                 criterion_local_restart=NelderAndMeadTermination(xtol=1e-2, ftol=1e-2),
                 n_max_local_restarts=0,
                 **kwargs):

        super().__init__(**kwargs)

        # the function to return the parameter
        self.func_params = func_params

        # the attributes for the simplex operations
        self.alpha, self.beta, self.gamma, self.delta = None, None, None, None

        # the scaling used for the initial simplex
        self.simplex_scaling = None

        # the specified termination criterion
        self.termination = termination

        # the initial point to be used to build the simplex
        self.x0 = x0
        self.opt = None

        self.func_display_attrs = disp_single_objective

        # everything needed for local restarts
        self.n_max_local_restarts = n_max_local_restarts
        self.criterion_local_restart = criterion_local_restart

        # internal variables to keep track of restarts
        self.restarts_disabled = False
        self.restart_history = []

    def _next(self):

        # perform step of nelder and mead algorithm and sort by F
        self.pop = self._step()
        self.opt = self.pop[0]

        # -------------------------------------------------------------------------------------------
        # Local Restarts (if n_max_restarts > 0)
        # -------------------------------------------------------------------------------------------

        # if a restart should be considered
        if self.n_max_local_restarts and not self.restarts_disabled:

            # if there should be a restart
            if self.criterion_local_restart.do_restart(self):

                # if there were at least 2 restarts
                not_enough_restarts = len(self.restart_history) <= 1

                if not not_enough_restarts:
                    improvement_to_last = self.restart_history[-1].get("F").min() - self.pop.get("F").min() > 1e-8

                # if all criteria for restarts are met
                if (not_enough_restarts or improvement_to_last) and len(
                        self.restart_history) < self.n_max_local_restarts:

                    # append to the restart history
                    self.restart_history.append(self.pop)

                    # work on a copy from now on
                    pop = self.pop.copy()

                    # initialize new simplex around best point and sort again
                    pop[1:].set("X", self.initialize_simplex(pop[0].X))
                    pop[1:] = self.evaluator.eval(self.problem, pop[1:])

                    # sort by F values
                    pop = pop[np.argsort(pop.get("F")[:, 0])]

                    # set the current population to the copy
                    self.pop = pop

                # otherwise just focus on the best result found so far until converging
                else:

                    # if restarted before - search for the best one
                    if len(self.restart_history) > 1:
                        best_F = [restart.get("F").min() for restart in self.restart_history]
                        I = np.array(best_F).argmin()
                        self.pop = self.restart_history[I]

                    # disable restarts for future
                    self.restarts_disabled = True

        return self.pop

    def _initialize(self):

        # initialize the function parameters
        self.alpha, self.beta, self.gamma, self.delta = self.func_params(self.problem)

        # reset the restart history
        self.restart_history = []

        # initialize the point
        if self.x0 is None:
            if not self.problem.has_bounds():
                raise Exception("Either provide an x0 or a problem with variable bounds!")

            # initialize randomly and make sure 5% is left for creating the initial simplex
            X = np.random.random(self.problem.n_var)
            self.x0 = denormalize(X, self.problem.xl, self.problem.xu)

        # parse the initial population from array or population object
        pop = pop_from_array_or_individual(self.x0)

        # if the simplex has not the correct number of points
        if len(pop) == 1:

            # the corresponding x values
            x0 = pop[0].X

            # if lower and upper bounds are given take 5% of the range and add
            if self.problem.has_bounds():
                self.simplex_scaling = 0.05 * (self.problem.xu - self.problem.xl)

            # no bounds are given do it based on x0 - MATLAB procedure
            else:
                self.simplex_scaling = 0.05 * x0
                # some value needs to be added if x0 is zero
                self.simplex_scaling[self.simplex_scaling == 0] = 0.00025

            # initialize the simplex
            X = self.initialize_simplex(x0)

            # create a population object
            pop = pop.merge(Population().new("X", X))

            # evaluate the values that are not already evaluated
            evaluate_if_not_done_yet(self.evaluator, self.problem, pop, algorithm=self)

        elif len(pop) != self.problem.n_var + 1:

            raise Exception("Provided initial population has size of %s, but should have size of %s" %
                            (len(pop), self.problem.n_var + 1))

        # sort by its corresponding function values
        self.pop = pop[np.argsort(pop.get("F")[:, 0])]
        self.opt = self.pop[0]

    def _step(self):

        # number of variables increased by one - matches equations in the paper
        pop, n = self.pop, self.problem.n_var - 1

        # calculate the centroid
        centroid = pop[:n + 1].get("X").mean(axis=0)

        # -------------------------------------------------------------------------------------------
        # REFLECT
        # -------------------------------------------------------------------------------------------

        # check the maximum alpha until the bounds are hit
        max_alpha = max_expansion_factor(centroid, (centroid - pop[n + 1].X), self.problem.xl, self.problem.xu)

        # reflect the point, consider factor if bounds are there, make sure in bounds (floating point) evaluate
        x_reflect = centroid + min(self.alpha, max_alpha) * (centroid - pop[n + 1].X)
        x_reflect = repair_out_of_bounds(self.problem, x_reflect)
        reflect = self.evaluator.eval(self.problem, Individual(X=x_reflect), algorithm=self)

        # whether a shrink is necessary or not - decided during this step
        shrink = False

        # if the new point is not better than the best, but better than second worst - just take it
        if pop[0].F <= reflect.F < pop[n].F:
            pop[n + 1] = reflect

        # if even better than the best - check for expansion
        elif reflect.F < pop[0].F:

            # -------------------------------------------------------------------------------------------
            # EXPAND
            # -------------------------------------------------------------------------------------------

            # the maximum expansion until the bounds are hit
            max_beta = max_expansion_factor(centroid, (x_reflect - centroid), self.problem.xl, self.problem.xu)

            # expand using the factor, consider bounds, make sure in case of floating point issues
            x_expand = centroid + min(self.beta, max_beta) * (x_reflect - centroid)
            x_expand = repair_out_of_bounds(self.problem, x_expand)

            # if the expansion is almost equal to reflection (if boundaries were hit) - no evaluation
            if np.allclose(x_expand, x_reflect, atol=1e-16):
                expand = reflect
            else:
                expand = self.evaluator.eval(self.problem, Individual(X=x_expand), algorithm=self)

            # if the expansion further improved take it - otherwise use expansion
            if expand.F < reflect.F:
                pop[n + 1] = expand
            else:
                pop[n + 1] = reflect

        # if not worst than the worst - outside contraction
        elif pop[n].F <= reflect.F < pop[n + 1].F:

            # -------------------------------------------------------------------------------------------
            # Outside Contraction
            # -------------------------------------------------------------------------------------------

            x_contract_outside = centroid + self.gamma * (x_reflect - centroid)
            contract_outside = self.evaluator.eval(self.problem, Individual(X=x_contract_outside), algorithm=self)

            if contract_outside.F <= reflect.F:
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

            if contract_inside.F < pop[n + 1].F:
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

        # finally sort the population by objective values
        pop = pop[np.argsort(pop.get("F")[:, 0])]

        return pop

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


# =========================================================================================================
# Interface
# =========================================================================================================

def nelder_mead(**kwargs):
    return NelderMead(**kwargs)
