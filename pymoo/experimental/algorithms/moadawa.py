import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.factory import get_decomposition
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================


class MOEADAWA(GeneticAlgorithm):
    """

    Adapted from
    Qi, Y., Ma, X., Liu, F., Jiao, L., Sun, J., & Wu, J. (2014). MOEA/D with Adaptive Weight Adjustment.
    Evolutionary Computation, 22(2), 231–264. https://doi.org/10.1162/EVCO_a_00109

    Implementation inspired from the MOEAD-AWA implementation in Matlab of the PlatEMO library.

    """

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition=Tchebicheff(),
                 prob_neighbor_mating=0.9,
                 rate_update_weight=0.05,
                 rate_evol=0.8,
                 wag=20,
                 archive_size_multiplier=1.5,
                 use_new_ref_dirs_initialization=True,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        MOEAD-AWA Algorithm.

        Parameters
        ----------
        ref_dirs
        n_neighbors
        decomposition
        prob_neighbor_mating
        rate_update_weight
        rate_evol
        wag
        archive_size_multiplier
        use_new_ref_dirs_initialization
        display
        kwargs
        """

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomp = decomposition

        self.rate_update_weight = rate_update_weight
        self.rate_evol = rate_evol
        self.wag = wag

        self.EP = None
        self.nEP = np.ceil(len(ref_dirs) * archive_size_multiplier)

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SBX(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PM(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs
        if use_new_ref_dirs_initialization:
            self.ref_dirs = 1.0 / (self.ref_dirs + 1e-6)
            self.ref_dirs = self.ref_dirs / np.sum(self.ref_dirs, axis=1)[:, None]

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        self.nds = NonDominatedSorting()

        # compute neighbors of reference directions using euclidean distance
        self._update_neighbors()

    def _update_neighbors(self):
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

    def _setup(self, problem, **kwargs):
        assert not problem.has_constraints(), "This implementation of MOEAD-AWA does not support any constraints."

        if isinstance(self.decomp, str):
            # for one or two objectives use tchebi otherwise pbi
            if self.decomp == 'auto':
                if self.problem.n_obj <= 2:
                    from pymoo.decomposition.tchebicheff import Tchebicheff
                    self.decomp = Tchebicheff()
                else:
                    from pymoo.decomposition.pbi import PBI
                    self.decomp = PBI()

    def _initialize_advance(self, infills=None, **kwargs):
        self.ideal_point = np.min(infills.get("F"), axis=0)
        self.EP = Population()

    def _infill(self):
        pass

    def _advance(self, **kwargs):
        repair, crossover, mutation = self.repair, self.mating.crossover, self.mating.mutation

        # retrieve the current population
        pop = self.pop

        offsprings = Population(len(pop))

        # iterate for each member of the population in random order
        for i in np.random.permutation(len(pop)):

            # all neighbors of this individual and corresponding weights
            N = self.neighbors[i, :]

            if np.random.random() < self.prob_neighbor_mating:
                parents = N[np.random.permutation(self.n_neighbors)]
            else:
                parents = np.random.permutation(self.pop_size)

            # do recombination and create an offspring
            off = crossover.do(self.problem, pop, parents[None, :crossover.n_parents], algorithm=self)
            off = mutation.do(self.problem, off, algorithm=self)
            off = off[np.random.randint(0, len(off))]

            # repair first in case it is necessary - disabled if instance of NoRepair
            off = repair.do(self.problem, off, algorithm=self)

            # evaluate the offspring
            self.evaluator.eval(self.problem, off, algorithm=self)

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)

            FV = self.decomp.do(pop[parents].get("F"),weights=self.ref_dirs[parents], ideal_point=self.ideal_point)
            off_FV = self.decomp.do(off.F[None, :], weights=self.ref_dirs[parents], ideal_point=self.ideal_point)
            I = np.where(off_FV < FV)[0]  # find locations where the offspring is strictly better than a parent
            pop[parents[I]] = off  # replace these parents by the offspring
            offsprings[i] = off  # save the offspring for the eventual external archive update

        # AWA specific part - update elite archive and reference directions
        # if no max generation is set then the behavior is always AWA
        max_gen = self.termination.__dict__.get("n_max_gen", 0)
        if self.n_gen >= self.rate_evol * max_gen and self.n_gen % self.wag == 0:
            # update external archive that should be bigger than population size
            self.EP = update_ep(self.EP, offsprings, self.nEP, self.nds)

            # remove a fraction of the crowded weights vectors and add new ones using the elite archive EP
            self.pop, self.ref_dirs = update_weight(self.pop,
                                                    self.ref_dirs,
                                                    # the small epsilon prevent NaNs although they won't stay anyway
                                                    self.ideal_point - 10e-7,
                                                    self.EP,
                                                    self.rate_update_weight * len(self.pop))

            # update the neighbor locations, it was not done in the PlatEMO implementation but it is in the paper
            self._update_neighbors()


def update_ep(EP, Offsprings, nEP, nds):
    """
    Update the external population archive
    """

    # merge population and keep only the first non-dominated front
    EP = Population.merge(EP, Offsprings)
    EP = EP[nds.do(EP.get("F"), only_non_dominated_front=True)]
    N, M = EP.get("F").shape

    # Delete the overcrowded solutions
    D = cdist(EP.get("F"), EP.get("F"))

    # prevent selection of a point with itself
    D[np.eye(len(D), dtype=np.bool)] = np.inf

    removed = np.zeros(N, dtype=np.bool)
    while sum(removed) < N - nEP:
        remain = np.flatnonzero(~removed)
        subDis = np.sort(D[np.ix_(remain, remain)], axis=1)

        # compute viscinity distance among the closest neighbors
        prodDist = np.prod(subDis[:, 0: min(M, len(remain))], axis=1)

        # select the point with the smallest viscinity distance to its neighbors and set it as removed
        worst = np.argmin(prodDist)
        removed[remain[worst]] = True

    return EP[~removed]


def update_weight(pop, W, Z, EP, nus):
    """
    Delete crowded weight vectors and add new ones
    """

    N, M = pop.get("F").shape

    # Update the current population by EP
    # Calculate the function value of each solution in Population or EP on each subproblem in W
    combined = Population.merge(pop, EP)
    combinedF = np.abs(combined.get("F") - Z)
    g = np.zeros((len(combined), W.shape[0]))
    for i in range(W.shape[0]):
        g[:, i] = np.max(combinedF * W[i, :], axis=1)

    # Choose the best solution for each subproblem
    best = np.argmin(g, axis=0)
    pop = combined[best]

    ######################################
    # Delete the overcrowded subproblems #
    ######################################

    D = cdist(pop.get("F"), pop.get("F"))

    # avoid selection of solutions with themselves
    D[np.eye(len(D), dtype=np.bool)] = np.inf

    deleted = np.zeros(len(pop), dtype=np.bool)
    while np.sum(deleted) < min(nus, len(EP)):
        remain = np.flatnonzero(~deleted)
        subD = D[np.ix_(remain, remain)]
        subDis = np.sort(subD, axis=1)

        # use viscinity distance to find the most crowded vector among the remaining ones and set it to be deleted
        worst = np.argmin(np.prod(subDis[:, 0:min(M, len(remain))], axis=1))
        deleted[remain[worst]] = True

    pop = pop[~deleted]
    W = W[~deleted, :]

    ######################################
    #       Add new subproblems          #
    ######################################

    # Determine the new solutions be added
    combined = Population.merge(pop, EP)
    selected = np.zeros(len(combined), dtype=np.bool)
    selected[:len(pop)] = True  # keep all solutions from pop and add solutions from EP
    D = cdist(combined.get("F"), combined.get("F"))
    D[np.eye(len(D), dtype=np.bool)] = np.inf
    while np.sum(selected) < min(N, len(selected)):
        # get the farthest solutions from already selected solutions using viscinity distance and select it
        subDis = np.sort(D[np.ix_(~selected, selected)], axis=1)
        best = np.argmax(np.prod(subDis[:, 0:min(M, subDis.shape[1])], axis=1))
        remain = np.flatnonzero(~selected)
        selected[remain[best]] = True

    # Add new subproblems to W
    newF = EP[selected[len(pop):]].get("F")

    # transform the weights using the WS transformation described in the paper
    # we don't care about NaNs as they will be eliminated later anyway
    with np.errstate(divide="ignore", invalid='ignore'):
        temp = 1. / (newF - Z)
        W = np.vstack([W, temp / np.sum(temp, axis=1)[:, None]])

    # Add new solutions
    pop = combined[selected]
    return pop, W
