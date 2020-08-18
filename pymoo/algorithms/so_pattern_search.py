import numpy as np

from pymoo.algorithms.so_local_search import LocalSearch
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.model.replacement import is_better
from pymoo.model.termination import Termination
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Implementation
# =========================================================================================================


class PatternSearchDisplay(SingleObjectiveDisplay):

    def __init__(self, **kwargs):
        super().__init__(favg=False, **kwargs)

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("delta", np.max(np.abs(algorithm.explr_delta)))


class PatternSearchTermination(Termination):

    def __init__(self, eps=1e-5, **kwargs):
        super().__init__()
        self.default = SingleObjectiveDefaultTermination(**kwargs)
        self.eps = eps

    def do_continue(self, algorithm):
        decision_default = self.default.do_continue(algorithm)
        delta = np.max(np.abs(algorithm.explr_delta))
        if delta < self.eps:
            return decision_default
        else:
            return True


class PatternSearch(LocalSearch):
    def __init__(self,
                 explr_delta=0.25,
                 explr_rho=0.5,
                 pattern_step=2,
                 eps=1e-5,
                 display=PatternSearchDisplay(),
                 **kwargs):
        """
        An implementation of well-known Hooke and Jeeves Pattern Search.

        Parameters
        ----------

        x0 : numpy.array
            The initial value where the local search should be initiated. If not provided `n_sample_points` are created
            created using latin hypercube sampling and the best solution found is set to `x0`.

        n_sample_points : int
            Number of sample points to be used to determine the initial search point. (Only used of `x0` is not provided)

        explr_delta : float
            The `delta` values which is used for the exploration move. If lower and upper bounds are provided the
            value is in relation to the overall search space. For instance, a value of 0.25 means that initially the
            pattern is created in 25% distance of the initial search point.

        explr_rho : float
            If the move was unsuccessful then the `delta` value is reduced by multiplying it with the value provided.
            For instance, `explr_rho` implies that with a value of `delta/2` is continued.

        pattern_step : float
            After the exploration move the new center is determined by following a promising direction.
            This value defines how large to step on this direction will be.

        eps : float
            This value is used for an additional termination criterion. When all delta values (maximum move along each variable)
            is less than epsilon the algorithm is terminated. Otherwise, the default termination criteria are also used.

        """

        super().__init__(display=display, **kwargs)
        self.explr_rho = explr_rho
        self.pattern_step = pattern_step
        self.explr_delta = explr_delta
        self.default_termination = PatternSearchTermination(eps=eps, x_tol=1e-6, f_tol=1e-6, nth_gen=1, n_last=30)

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)

        # make delta a vector - the sign is later updated individually
        if not isinstance(self.explr_delta, np.ndarray):
            self.explr_delta = np.ones(self.problem.n_var) * self.explr_delta

    def _next(self):

        # in the beginning of each iteration first do an exploration move
        self._previous = self.opt[0]
        self._current = self._exploration_move(self._previous)

        # one iteration is the combination of this two moves repeatedly until delta needs to be reduced
        while self._previous != self._current:

            # use the pattern move to get a new trial vector
            trial = self._pattern_move(self._previous, self._current)

            # perform an exploration move around the trial vector - the best known solution is always stored in _current
            explr = self._exploration_move(trial, opt=self._current)

            # we can break if we did not improve
            if not is_better(explr, self._current, eps=1e-6):
                break

            # else also check if we are terminating - otherwise this loop might run far too long
            self._set_optimum()
            if not self.termination.do_continue(self):
                break

            self._previous, self._current = self._current, explr

        self.explr_delta *= self.explr_rho

    def _pattern_move(self, _current, _next):

        # get the direction and assign the corresponding delta value
        direction = (_next.X - _current.X)

        # get the delta sign adjusted
        sign = np.sign(direction)
        sign[sign == 0] = -1
        self.explr_delta = sign * np.abs(self.explr_delta)

        # calculate the new X and repair out of bounds if necessary
        X = _current.X + self.pattern_step * direction
        set_to_bounds_if_outside_by_problem(self.problem, X)

        # create the new center individual without evaluating it
        trial = Individual(X=X)

        return trial

    def _exploration_move(self, center, opt=None):
        if opt is None:
            opt = center

        def step(x, delta, k):

            # copy and add delta to the new point
            X = np.copy(x)

            # normalize the delta by the bounds if they are provided by the problem
            eps = delta[k]

            # if the problem has bounds normalize the delta
            if self.problem.has_bounds():
                xl, xu = self.problem.bounds()
                eps *= (xu[k] - xl[k])

            # now add to the current solution
            X[k] = X[k] + eps

            # repair if out of bounds if necessary
            X = set_to_bounds_if_outside_by_problem(self.problem, X)

            # return the new solution as individual
            mutant = pop_from_array_or_individual(X)[0]

            return mutant

        for k in range(self.problem.n_var):

            # create the the individual and evaluate it
            mutant = step(center.X, self.explr_delta, k)
            self.evaluator.eval(self.problem, mutant, algorithm=self)
            self.pop = Population.merge(self.pop, mutant)

            if is_better(mutant, opt):
                center, opt = mutant, mutant

            else:

                # inverse the sign of the delta
                self.explr_delta[k] = - self.explr_delta[k]

                # now try the other sign if there was no improvement
                mutant = step(center.X, self.explr_delta, k)
                self.evaluator.eval(self.problem, mutant, algorithm=self)
                self.pop = Population.merge(self.pop, mutant)

                if is_better(mutant, opt):
                    center, opt = mutant, mutant

        return opt


parse_doc_string(PatternSearch.__init__)
