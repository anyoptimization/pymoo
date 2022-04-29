import numpy as np

from pymoo.algorithms.base.local import LocalSearch
from pymoo.docs import parse_doc_string
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.replacement import is_better
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.optimum import filter_optimum


# =========================================================================================================
# Implementation
# =========================================================================================================


class PatternSearchDisplay(SingleObjectiveDisplay):

    def __init__(self, **kwargs):
        super().__init__(favg=False, **kwargs)

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("rho", algorithm._rho)


class PatternSearch(LocalSearch):
    def __init__(self,
                 init_delta=0.25,
                 rho=0.5,
                 step_size=1.0,
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

        delta : float
            The `delta` values which is used for the exploration move. If lower and upper bounds are provided the
            value is in relation to the overall search space. For instance, a value of 0.25 means that initially the
            pattern is created in 25% distance of the initial search point.

        rho : float
            If the move was unsuccessful then the `delta` value is reduced by multiplying it with the value provided.
            For instance, `explr_rho` implies that with a value of `delta/2` is continued.

        step_size : float
            After the exploration move the new center is determined by following a promising direction.
            This value defines how large to step on this direction will be.

        """

        super().__init__(display=display, **kwargs)
        self.rho = rho
        self.step_size = step_size
        self.init_delta = init_delta

        self._rho = rho
        self._delta = None
        self._center = None
        self._current = None
        self._trial = None
        self._direction = None
        self._sign = None

    def _setup(self, problem, **kwargs):

        if problem.has_bounds():
            xl, xu = problem.bounds()
            self._delta = self.init_delta * (xu - xl)
        else:
            self._delta = np.abs(self.x0) / 2.0
            self._delta[self._delta <= 1.0] = 1.0

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self._center, self._explr = self.x0, None
        self._sign = np.ones(self.problem.n_var)

    def _local_advance(self, **kwargs):

        # this happens only in the first iteration
        if self._explr is None:
            self._explr = self._exploration_move(self._center)

        else:
            # whether the last iteration has resulted in a new optimum
            has_improved = is_better(self._explr, self._center, eps=0.0)

            # that means that the exploration did not found any new point and was thus unsuccessful
            if not has_improved:

                # keep track of the rho values in the normalized space
                self._rho = self._rho * self.rho

                # explore around the current center to try finding a suitable direction
                self._explr = self._exploration_move(self._center)

            # if we have found a direction in the last iteration to be worth following
            else:

                # get the direction which was successful in the last move
                self._direction = (self._explr.X - self._center.X)

                # declare the exploration point the new center
                self._center = self._explr

                # use the pattern move to get a new trial vector along that given direction
                self._trial = self._pattern_move(self._center, self._direction)

                # get the delta sign adjusted for the exploration
                self._sign = calc_sign(self._direction)

                # explore around the current center to try finding a suitable direction
                self._explr = self._exploration_move(self._trial)

        self.pop = Population.create(self._center, self._explr)

    def _exploration_move(self, center):

        # randomly iterate over all variables
        for k in range(self.problem.n_var):

            # the value to be tried first is given by the amount times the sign
            _delta = self._sign[k] * self._rho * self._delta

            # make a step of delta on the k-th variable
            _explr = step(self.problem, center.X, _delta, k)
            self.evaluator.eval(self.problem, _explr, algorithm=self)

            if is_better(_explr, center, eps=0.0):
                center = _explr

            # if not successful try the other direction
            else:

                # now try the negative value of delta and see if we can improve
                _explr = step(self.problem, center.X, -1 * _delta, k)
                self.evaluator.eval(self.problem, _explr, algorithm=self)

                if is_better(_explr, center, eps=0.0):
                    center = _explr

        return center

    def _pattern_move(self, _current, _direction):

        # calculate the new X and repair out of bounds if necessary
        X = _current.X + self.step_size * _direction
        set_to_bounds_if_outside_by_problem(self.problem, X)

        # create the new center individual and evaluate it
        trial = Individual(X=X)
        self.evaluator.eval(self.problem, trial, algorithm=self)

        return trial

    def _set_optimum(self):
        pop = self.pop if self.opt is None else Population.merge(self.opt, self.pop)
        self.opt = filter_optimum(pop, least_infeasible=True)


def calc_sign(direction):
    sign = np.sign(direction)
    sign[sign == 0] = -1
    return sign


def step(problem, x, delta, k):
    # copy and add delta to the new point
    X = np.copy(x)

    # if the problem has bounds normalize the delta
    # if problem.has_bounds():
    #     xl, xu = problem.bounds()
    #     delta *= (xu[k] - xl[k])

    # now add to the current solution
    X[k] = X[k] + delta[k]

    # repair if out of bounds if necessary
    X = set_to_bounds_if_outside_by_problem(problem, X)

    return Individual(X=X)


parse_doc_string(PatternSearch.__init__)
