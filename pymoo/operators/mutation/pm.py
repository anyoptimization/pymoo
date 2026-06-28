"""Polynomial mutation operator."""

import numpy as np

from pymoo.core.mutation import Mutation
from pymoo.core.variable import get, Real
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.util import default_random_state


# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------


@default_random_state
def mut_pm(X, xl, xu, eta, prob, at_least_once, random_state=None):
    """Apply polynomial mutation to a population.

    Args:
        X: Population variables of shape (n, n_var).
        xl: Lower bounds for variables.
        xu: Upper bounds for variables.
        eta: Distribution indices (eta values) of shape (n,).
        prob: Mutation probabilities per variable of shape (n_var,).
        at_least_once: Whether to mutate at least one variable per individual.
        random_state: Random state for reproducibility.

    Returns:
        Mutated population of same shape as X.
    """
    n, n_var = X.shape
    assert len(eta) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = mut_binomial(n, n_var, prob, at_least_once=at_least_once, random_state=random_state)
    mut[:, xl == xu] = False

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]

    X = X[mut]
    eta = np.tile(eta[:, None], (1, n_var))[mut]

    delta1 = (X - _xl) / (_xu - _xl)
    delta2 = (_xu - X) / (_xu - _xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand = random_state.random(X.shape)
    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    with np.errstate(invalid="ignore"):
        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (_xu - _xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    # set the values for output
    Xp[mut] = _Y

    # in case out of bounds repair (very unlikely)
    Xp = set_to_bounds_if_outside(Xp, xl, xu)

    return Xp


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------


class PolynomialMutation(Mutation):
    """Polynomial mutation operator for continuous variables."""

    def __init__(self, prob=0.9, eta=20, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))

    def _do(self, problem, X, params=None, *args, random_state=None, **kwargs):  # noqa: D417
        """Perform polynomial mutation.

        Args:
            problem: The optimization problem.
            X: Population variables.
            params: Additional parameters.
            random_state: Random state for reproducibility.

        Returns:
            Mutated population.
        """
        X = X.astype(float)

        eta = get(self.eta, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))

        Xp = mut_pm(
            X,
            problem.xl,
            problem.xu,
            eta,
            prob_var,
            at_least_once=self.at_least_once,
            random_state=random_state,
        )

        return Xp


class PM(PolynomialMutation):
    """Alias for PolynomialMutation."""

    pass
