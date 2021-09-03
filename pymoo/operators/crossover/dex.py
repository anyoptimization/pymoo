import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.model.population import Population
from pymoo.operators.repair.bounds_repair import is_out_of_bounds_by_problem


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


def de_repair_random_init(Xp, X, xl, xu):
    XL = xl[None, :].repeat(len(Xp), axis=0)
    XU = xu[None, :].repeat(len(Xp), axis=0)

    i, j = np.where(Xp < XL)
    if len(i) > 0:
        Xp[i, j] = XL[i, j] + np.random.random(len(i)) * (X[i, j] - XL[i, j])

    i, j = np.where(Xp > XU)
    if len(i) > 0:
        Xp[i, j] = XU[i, j] - np.random.random(len(i)) * (XU[i, j] - X[i, j])


class DifferentialEvolutionCrossover(Crossover):

    def __init__(self,
                 weight=0.85,
                 dither=None,
                 jitter=False,
                 n_diffs=1,
                 **kwargs):
        super().__init__(1 + 2 * n_diffs, 1, **kwargs)
        self.n_diffs = n_diffs
        self.weight = weight
        self.dither = dither
        self.jitter = jitter

    def do(self, problem, pop, parents, **kwargs):
        X = pop.get("X")[parents.T].copy()
        assert len(X.shape) == 3, "Please provide a three-dimensional matrix n_parents x pop_size x n_vars."

        n_parents, n_matings, n_var = X.shape

        # a mask over matings that need to be repeated
        m = np.arange(n_matings)

        # if the user provides directly an f value to use
        F = self.weight if self.weight is not None else np.random.uniform(low=0.5, high=1.0, size=len(m))

        # prepare the out to be set
        Xp = de_differential(X[:, m], F)

        # if the problem has boundaries to be considered
        if problem.has_bounds():

            for k in range(20):
                # find the individuals which are still infeasible
                m = is_out_of_bounds_by_problem(problem, Xp)

                F = np.random.uniform(low=0.5, high=1.0, size=len(m))

                # actually execute the differential equation
                Xp[m] = de_differential(X[:, m], F)

            # if still infeasible do a random initialization
            de_repair_random_init(Xp, X[0], *problem.bounds())

        return Population.new("X", Xp)
