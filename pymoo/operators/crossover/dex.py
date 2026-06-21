import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.expx import mut_exp
from pymoo.operators.repair.bounds_repair import is_out_of_bounds_by_problem, repair_random_init
from pymoo.util import default_random_state


@default_random_state
def de_differential(X, F, dither=None, jitter=True, gamma=0.0001, return_differentials=False, random_state=None):
    n_parents, n_matings, n_var = X.shape
    assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

    # make sure F is a one-dimensional vector
    F = np.ones(n_matings) * F

    # build the pairs for the differentials
    pairs = (np.arange(n_parents - 1) + 1).reshape(-1, 2)

    # the differentials from each pair subtraction
    diffs = np.zeros((n_matings, n_var))

    # for each difference
    for i, j in pairs:

        if dither == "vector":
            F = (F + random_state.random(n_matings) * (1 - F))
        elif dither == "scalar":
            F = F + random_state.random() * (1 - F)

        # http://www.cs.ndsu.nodak.edu/~siludwig/Publish/papers/SSCI20141.pdf
        if jitter:
            F = (F * (1 + gamma * (random_state.random(n_matings) - 0.5)))

        # an add the difference to the first vector
        diffs += F[:, None] * (X[i] - X[j])

    # now add the differentials to the first parent
    Xp = X[0] + diffs

    if return_differentials:
        return Xp, diffs
    else:
        return Xp


class DEX(Crossover):

    def __init__(self,
                 F=None,
                 CR=0.7,
                 variant="bin",
                 dither=None,
                 jitter=False,
                 n_diffs=1,
                 n_iter=1,
                 at_least_once=True,
                 **kwargs):

        super().__init__(1 + 2 * n_diffs, 1, **kwargs)
        self.n_diffs = n_diffs
        self.F = F
        self.CR = CR
        self.variant = variant
        self.at_least_once = at_least_once
        self.dither = dither
        self.jitter = jitter
        self.n_iter = n_iter

    def do(self, problem, pop, parents=None, *args, random_state, **kwargs):

        # if a parents with array with mating indices is provided -> transform the input first
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        # get the actual values from each of the parents
        X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1).copy()

        n_parents, n_matings, n_var = X.shape

        # a mask over matings that need to be repeated
        m = np.arange(n_matings)

        # if the user provides directly an F value to use
        F = self.F if self.F is not None else rnd_F(m, random_state=random_state)

        # prepare the out to be set
        Xp = de_differential(X[:, m], F, random_state=random_state)

        # if the problem has boundaries to be considered
        if problem.has_bounds():

            for k in range(self.n_iter):
                # find the individuals which are still infeasible
                m = is_out_of_bounds_by_problem(problem, Xp)

                F = rnd_F(m, random_state=random_state)

                # actually execute the differential equation
                Xp[m] = de_differential(X[:, m], F, random_state=random_state)

            # if still infeasible do a random initialization
            Xp = repair_random_init(Xp, X[0], *problem.bounds())

        if self.variant == "bin":
            M = mut_binomial(n_matings, n_var, self.CR, at_least_once=self.at_least_once, random_state=random_state)
        elif self.variant == "exp":
            M = mut_exp(n_matings, n_var, self.CR, at_least_once=self.at_least_once, random_state=random_state)
        else:
            raise Exception(f"Unknown variant: {self.variant}")

        # take the first parents (this is already a copy)
        X = X[0]

        # set the corresponding values from the donor vector
        X[M] = Xp[M]

        return Population.new("X", X)


@default_random_state
def rnd_F(m, random_state=None):
    return 0.5 * (1 + random_state.uniform(size=len(m)))


# =========================================================================================================
# DE Repair strategies
# =========================================================================================================

def bounce_back(X, Xb, xl, xu, random_state=None):
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)
    i, j = np.where(X < XL)
    if len(i) > 0:
        rs = random_state if random_state is not None else np.random
        X[i, j] = XL[i, j] + rs.random(len(i)) * (Xb[i, j] - XL[i, j])
    i, j = np.where(X > XU)
    if len(i) > 0:
        rs = random_state if random_state is not None else np.random
        X[i, j] = XU[i, j] - rs.random(len(i)) * (XU[i, j] - Xb[i, j])
    return X


def midway(X, Xb, xl, xu, random_state=None):
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)
    i, j = np.where(X < XL)
    if len(i) > 0:
        X[i, j] = XL[i, j] + (Xb[i, j] - XL[i, j]) / 2
    i, j = np.where(X > XU)
    if len(i) > 0:
        X[i, j] = XU[i, j] - (XU[i, j] - Xb[i, j]) / 2
    return X


def to_bounds(X, Xb, xl, xu, random_state=None):
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)
    i, j = np.where(X < XL)
    if len(i) > 0:
        X[i, j] = XL[i, j]
    i, j = np.where(X > XU)
    if len(i) > 0:
        X[i, j] = XU[i, j]
    return X


def rand_init(X, Xb, xl, xu, random_state=None):
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)
    rs = random_state if random_state is not None else np.random
    i, j = np.where(X < XL)
    if len(i) > 0:
        X[i, j] = XL[i, j] + rs.random(len(i)) * (XU[i, j] - XL[i, j])
    i, j = np.where(X > XU)
    if len(i) > 0:
        X[i, j] = XU[i, j] - rs.random(len(i)) * (XU[i, j] - XL[i, j])
    return X


DE_REPAIRS = {
    "bounce-back": bounce_back,
    "midway": midway,
    "rand-init": rand_init,
    "to-bounds": to_bounds,
}


# =========================================================================================================
# DEM — DE Mutation as a Crossover operator
# =========================================================================================================

class DEM(Crossover):
    """DE mutation operator (donor vector computation), usable as a standalone Crossover."""

    def __init__(self, F=None, gamma=1e-4, de_repair="bounce-back", n_diffs=1, **kwargs):
        if F is None:
            F = (0.0, 1.0)
        self.F = F
        self.gamma = gamma
        if callable(de_repair):
            self.de_repair = de_repair
        else:
            if de_repair not in DE_REPAIRS:
                raise KeyError(f"de_repair must be callable or one of {list(DE_REPAIRS.keys())}")
            self.de_repair = DE_REPAIRS[de_repair]
        super().__init__(1 + 2 * n_diffs, 1, prob=1.0, **kwargs)

    def do(self, problem, pop, parents=None, *args, random_state=None, **kwargs):
        if parents is not None:
            pop = pop[parents]

        # pop shape: [n_matings, n_parents] — swap to [n_parents, n_matings]
        Xr = np.swapaxes(pop, 0, 1).get("X")  # [n_parents, n_matings, n_var]
        n_parents, n_matings, n_var = Xr.shape
        assert n_parents % 2 == 1

        pairs = (np.arange(n_parents - 1) + 1).reshape(-1, 2)

        diffs = np.zeros((n_matings, n_var))
        for i, j in pairs:
            F = self._sample_F(n_matings, random_state)
            if self.gamma is not None:
                F = F[:, None] * (1 + self.gamma * (random_state.random((n_matings, n_var)) - 0.5))
                diffs += F * (Xr[i] - Xr[j])
            else:
                diffs += F[:, None] * (Xr[i] - Xr[j])

        V = Xr[0] + diffs

        if problem.has_bounds():
            V = self.de_repair(V, Xr[0], *problem.bounds(), random_state=random_state)

        return Population.new("X", V)

    def _sample_F(self, n, random_state):
        if hasattr(self.F, "__iter__"):
            lo, hi = self.F[0], self.F[1]
            return lo + random_state.random(n) * (hi - lo)
        return np.full(n, self.F)
