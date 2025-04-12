import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import vectorized_cdist


# ---------------------------------------------------------------------------------------------------------
# Environmental Survival (in the original paper it is referred to as archiving)
# ---------------------------------------------------------------------------------------------------------


class SPEA2Survival(Survival):

    def __init__(self, normalize=False, filter_infeasible=True):
        super().__init__(filter_infeasible)

        # whether the survival should considered normalized distance or just raw
        self.normalize = normalize

        # an object keeping track of normalization points
        self.norm = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the domination matrix to see the relation for each solution to another
        M = Dominator().calc_domination_matrix(F)

        # the number of solutions each individual dominates
        S = (M == 1).sum(axis=0)

        # the raw fitness of each solution - strength of its dominators
        R = ((M == -1) * S).sum(axis=1)

        # determine what k-th nearest neighbor to consider
        k = int(np.sqrt(len(pop)))
        if k >= len(pop):
            k = len(pop) - 1

        # if normalization is enabled keep track of ideal and nadir
        if self.normalize:

            # initialize the first time and then always update the boundary points
            if self.norm is None:
                self.norm = HyperplaneNormalization(F.shape[1])
            self.norm.update(F)

            ideal, nadir = self.norm.ideal_point, self.norm.nadir_point

            _F = (F - ideal) / (nadir - ideal)
            dists = vectorized_cdist(_F, _F, fill_diag_with_inf=True)

        # if no normalize is required simply use the F values from the population
        else:
            dists = vectorized_cdist(F, F, fill_diag_with_inf=True)

        # the distances sorted for each individual
        sdists = np.sort(dists, axis=1)

        # inverse distance as part of the fitness
        D = 1 / (sdists[:, k] + 2)

        # the actual fitness value used to determine who survives
        SPEA_F = R + D

        # set all the attributes to the population
        pop.set(SPEA_F=SPEA_F, SPEA_R=R, SPEA_D=D)

        # get all the non-dominated solutions
        survivors = list(np.where(np.all(M >= 0, axis=1))[0])

        # if we normalize give boundary points most importance - give the boundary points in the nds set the lowest fit.
        if self.normalize:
            I = vectorized_cdist(self.norm.extreme_points, F[survivors]).argmin(axis=1)
            pop[survivors][I].set("SPEA_F", -1.0)

        # identify the remaining individuals to choose from
        H = set(survivors)
        rem = np.array([k for k in range(len(pop)) if k not in H])

        # if not enough solutions, will up by F
        if len(survivors) < n_survive:

            # sort them by the fitness values (lower is better) and append them
            rem_by_F = rem[SPEA_F[rem].argsort()]
            survivors.extend(rem_by_F[:n_survive - len(survivors)])

        # if too many, delete based on distances
        elif len(survivors) > n_survive:

            # remove one individual per loop, until we hit n_survive
            while len(survivors) > n_survive:
                i = dists[survivors][:, survivors].min(axis=1).argmin()
                survivors = [survivors[j] for j in range(len(survivors)) if j != i]

        return pop[survivors]


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection
# ---------------------------------------------------------------------------------------------------------


def spea_binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f, = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:
            S[i] = compare(a, pop[a].get("SPEA_F"), b, pop[b].get("SPEA_F"), method='smaller_is_better',
                           return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------------------------------------


class SPEA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(spea_binary_tournament),
                 crossover=SBX(),
                 mutation=PM(),
                 survival=SPEA2Survival(normalize=True),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """

        SPEA2 - Strength Pareto EA 2

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)
        self.termination = DefaultMultiObjectiveTermination()


parse_doc_string(SPEA2.__init__)

