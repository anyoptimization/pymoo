import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.expx import mut_exp
from pymoo.operators.repair.bounds_repair import is_out_of_bounds_by_problem, repair_random_init


def de_differential(X, F, dither=None, jitter=True, gamma=0.0001, return_differentials=False):
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
            F = (F + np.random.random(n_matings) * (1 - F))
        elif dither == "scalar":
            F = F + np.random.random() * (1 - F)

        # http://www.cs.ndsu.nodak.edu/~siludwig/Publish/papers/SSCI20141.pdf
        if jitter:
            F = (F * (1 + gamma * (np.random.random(n_matings) - 0.5)))

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

    def do(self, problem, pop, parents=None, **kwargs):

        # if a parents with array with mating indices is provided -> transform the input first
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        # get the actual values from each of the parents
        X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1).copy()

        n_parents, n_matings, n_var = X.shape

        # a mask over matings that need to be repeated
        m = np.arange(n_matings)

        # if the user provides directly an F value to use
        F = self.F if self.F is not None else rnd_F(m)

        # prepare the out to be set
        Xp = de_differential(X[:, m], F)

        # if the problem has boundaries to be considered
        if problem.has_bounds():

            for k in range(self.n_iter):
                # find the individuals which are still infeasible
                m = is_out_of_bounds_by_problem(problem, Xp)

                F = rnd_F(m)

                # actually execute the differential equation
                Xp[m] = de_differential(X[:, m], F)

            # if still infeasible do a random initialization
            Xp = repair_random_init(Xp, X[0], *problem.bounds())

        if self.variant == "bin":
            M = mut_binomial(n_matings, n_var, self.CR, at_least_once=self.at_least_once)
        elif self.variant == "exp":
            M = mut_exp(n_matings, n_var, self.CR, at_least_once=self.at_least_once)
        else:
            raise Exception(f"Unknown variant: {self.variant}")

        # take the first parents (this is already a copy)
        X = X[0]

        # set the corresponding values from the donor vector
        X[M] = Xp[M]

        return Population.new("X", X)


def rnd_F(m):
    return 0.5 * (1 + np.random.uniform(size=len(m)))
