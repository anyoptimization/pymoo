import numpy as np

from pymoo.algorithms.base.local import LocalSearch
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.replacement import is_better
from pymoo.docs import parse_doc_string
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.optimum import filter_optimum
from pymoo.util import default_random_state


# =========================================================================================================
# Implementation
# =========================================================================================================


class PatternSearch(LocalSearch):
    def __init__(self,
                 init_delta=0.25,
                 init_rho=0.5,
                 step_size=1.0,
                 output=SingleObjectiveOutput(),
                 **kwargs):
        """
        An implementation of well-known Hooke and Jeeves Pattern Search.

        Parameters
        ----------

        x0 : numpy.array
            The initial value where the local search should be initiated. If not provided `n_sample_points` are
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

        super().__init__(output=output, **kwargs)
        self.init_rho = init_rho
        self.init_delta = init_delta
        self.step_size = step_size

        self.n_not_improved = 0

        self._rho = init_rho
        self._delta = None
        self._center = None
        self._current = None
        self._trial = None
        self._direction = None
        self._sign = None

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)
        self._center, self._explr = self.x0, self.x0
        self._sign = np.ones(self.problem.n_var)

        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            self._delta = self.init_delta * (xu - xl)
        else:
            self._delta = np.abs(self.x0.X) / 2.0
            self._delta[self._delta <= 1.0] = 1.0

    def _next(self):

        # whether the last iteration has resulted in a new optimum or not
        has_improved = is_better(self._explr, self._center)

        # that means that the exploration did not find any new point and was thus unsuccessful
        if not has_improved:

            # increase the counter (by default this will be initialized to 0 and directly increased to 1)
            self.n_not_improved += 1

            # keep track of the rho values in the normalized space
            self._rho = self.init_rho ** self.n_not_improved

            # explore around the current center - try finding a suitable direction
            self._explr = yield from exploration_move(self.problem, self._center, self._sign, self._delta, self._rho)

        # if we have found a direction in the last iteration to be worth following
        else:

            # get the direction which was successful in the last move
            self._direction = (self._explr.X - self._center.X)

            # declare the exploration point the new center (it has led to an improvement in the last iteration!)
            self._center = self._explr

            # use the pattern move to get a new trial vector along that given direction
            self._trial = yield pattern_move(self.problem, self._center, self._direction, self.step_size)

            # get the delta sign adjusted for the exploration
            self._sign = calc_sign(self._direction)

            # explore around the current center to try finding a suitable direction
            self._explr = yield from exploration_move(self.problem, self._trial, self._sign, self._delta, self._rho)

        self.pop = Population.create(self._center, self._explr)

    def _set_optimum(self):
        pop = self.pop if self.opt is None else Population.merge(self.opt, self.pop)
        self.opt = filter_optimum(pop, least_infeasible=True)


@default_random_state
def exploration_move(problem, center, sign, delta, rho, randomize=True, random_state=None):
    n_var = problem.n_var

    # the order for the variable iteration
    if randomize:
        K = random_state.permutation(n_var)
    else:
        K = np.arange(n_var)

    # iterate over each variable
    for k in K:

        # the value to be tried first is given by the amount times the sign
        _delta = sign[k] * rho * delta

        # make a step of delta on the k-th variable
        _explr = yield step_along_axis(problem, center.X, _delta, k)

        if is_better(_explr, center, eps=0.0):
            center = _explr

        # if not successful try the other direction
        else:

            # now try the negative value of delta and see if we can improve
            _explr = yield step_along_axis(problem, center.X, -1 * _delta, k)

            if is_better(_explr, center, eps=0.0):
                center = _explr

    return center


def pattern_move(problem, current, direction, step_size):
    # calculate the new X and repair out of bounds if necessary
    X = current.X + step_size * direction
    set_to_bounds_if_outside_by_problem(problem, X)

    # create the new center individual
    return Individual(X=X)


def calc_sign(direction):
    sign = np.sign(direction)
    sign[sign == 0] = -1
    return sign


def step_along_axis(problem, x, delta, axis):
    # copy and add delta to the new point
    X = np.copy(x)

    # now add to the current solution
    X[axis] = X[axis] + delta[axis]

    # repair if out of bounds if necessary
    X = set_to_bounds_if_outside_by_problem(problem, X)

    return Individual(X=X)


parse_doc_string(PatternSearch.__init__)
