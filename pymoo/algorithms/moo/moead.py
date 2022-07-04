import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.algorithm import LoopwiseAlgorithm
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.core.variable import Real, get
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.reference_direction import default_ref_dirs


class NeighborhoodSelection(Selection):

    def __init__(self, prob=1.0) -> None:
        super().__init__()
        self.prob = Real(prob, bounds=(0.0, 1.0))

    def _do(self, problem, pop, n_select, n_parents, neighbors=None, **kwargs):
        assert n_select == len(neighbors)
        P = np.full((n_select, n_parents), -1)

        prob = get(self.prob, size=n_select)

        for k in range(n_select):
            if np.random.random() < prob[k]:
                P[k] = np.random.choice(neighbors[k], n_parents, replace=False)
            else:
                P[k] = np.random.permutation(len(pop))[:n_parents]

        return P


# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEAD(LoopwiseAlgorithm, GeneticAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_neighbors=20,
                 decomposition=None,
                 prob_neighbor_mating=0.9,
                 sampling=FloatRandomSampling(),
                 crossover=SBX(prob=1.0, eta=20),
                 mutation=PM(prob_var=None, eta=20),
                 output=MultiObjectiveOutput(),
                 **kwargs):

        # reference directions used for MOEAD
        self.ref_dirs = ref_dirs

        # the decomposition metric used
        self.decomposition = decomposition

        # the number of neighbors considered during mating
        self.n_neighbors = n_neighbors

        self.neighbors = None

        self.selection = NeighborhoodSelection(prob=prob_neighbor_mating)

        super().__init__(pop_size=len(ref_dirs),
                         sampling=sampling,
                         crossover=crossover,
                         mutation=mutation,
                         eliminate_duplicates=NoDuplicateElimination(),
                         output=output,
                         advance_after_initialization=False,
                         **kwargs)

    def _setup(self, problem, **kwargs):
        assert not problem.has_constraints(), "This implementation of MOEAD does not support any constraints."

        # if no reference directions have been provided get them and override the population size and other settings
        if self.ref_dirs is None:
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

        # if the decomposition is not set yet, set the default
        if self.decomposition is None:
            self.decomposition = default_decomp(problem)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self.ideal = np.min(self.pop.get("F"), axis=0)

    def _next(self):
        pop = self.pop

        # iterate for each member of the population in random order
        for k in np.random.permutation(len(pop)):
            # get the parents using the neighborhood selection
            P = self.selection.do(self.problem, pop, 1, self.mating.crossover.n_parents, neighbors=[self.neighbors[k]])

            # perform a mating using the default operators - if more than one offspring just pick the first
            off = np.random.choice(self.mating.do(self.problem, pop, 1, parents=P, n_max_iterations=1))

            # evaluate the offspring
            off = yield off

            # update the ideal point
            self.ideal = np.min(np.vstack([self.ideal, off.F]), axis=0)

            # now actually do the replacement of the individual is better
            self._replace(k, off)

    def _replace(self, k, off):
        pop = self.pop

        # calculate the decomposed values for each neighbor
        N = self.neighbors[k]
        FV = self.decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal)
        off_FV = self.decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal)

        # this makes the algorithm to support constraints - not originally proposed though and not tested enough
        # if self.problem.has_constraints():
        #     CV, off_CV = pop[N].get("CV")[:, 0], np.full(len(off_FV), off.CV)
        #     fmax = max(FV.max(), off_FV.max())
        #     FV, off_FV = parameter_less(FV, CV, fmax=fmax), parameter_less(off_FV, off_CV, fmax=fmax)

        # get the absolute index in F where offspring is better than the current F (decomposed space)
        I = np.where(off_FV < FV)[0]
        pop[N[I]] = off


class ParallelMOEAD(MOEAD):

    def __init__(self, ref_dirs, **kwargs):
        super().__init__(ref_dirs, **kwargs)
        self.indices = None

    def _infill(self):
        pop_size, cross_parents, cross_off = self.pop_size, self.mating.crossover.n_parents, self.mating.crossover.n_offsprings

        # do the mating in a random order
        indices = np.random.permutation(len(self.pop))[:self.n_offsprings]

        # get the parents using the neighborhood selection
        P = self.selection.do(self.problem, self.pop, self.n_offsprings, cross_parents,
                              neighbors=self.neighbors[indices])

        # do not any duplicates elimination - thus this results in exactly pop_size * n_offsprings offsprings
        off = self.mating.do(self.problem, self.pop, 1e12, n_max_iterations=1, parents=P)

        # select a random offspring from each mating
        off = Population.create(*[np.random.choice(pool) for pool in np.reshape(off, (self.n_offsprings, -1))])

        # store the indices because of the neighborhood matching in advance
        self.indices = indices

        return off

    def _advance(self, infills=None, **kwargs):
        assert len(self.indices) == len(infills), "Number of infills must be equal to the one created beforehand."

        # update the ideal point before starting to replace
        self.ideal = np.min(np.vstack([self.ideal, infills.get("F")]), axis=0)

        # now do the replacements as in the loop-wise version
        for k, off in enumerate(infills):
            self._replace(self.indices[k], off)


def default_decomp(problem):
    if problem.n_obj <= 2:
        from pymoo.decomposition.tchebicheff import Tchebicheff
        return Tchebicheff()
    else:
        from pymoo.decomposition.pbi import PBI
        return PBI()


parse_doc_string(MOEAD.__init__)
