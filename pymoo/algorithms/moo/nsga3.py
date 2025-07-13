import warnings

import numpy as np
from numpy.linalg import LinAlgError

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.functions import load_function
from pymoo.util.misc import intersect, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util import default_random_state


# =========================================================================================================
# Implementation
# =========================================================================================================

@default_random_state
def comp_by_cv_then_random(pop, P, random_state=None, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = random_state.choice([a, b])

    return S[:, None].astype(int)


class NSGA3(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=SBX(eta=30, prob=1.0),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        pop_size : int (default = None)
            By default the population size is set to None which means that it will be equal to the number of reference
            line. However, if desired this can be overwritten by providing a positive number.
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        self.ref_dirs = ref_dirs

        # in case of R-NSGA-3 they will be None - otherwise this will be executed
        if self.ref_dirs is not None:

            if pop_size is None:
                pop_size = len(self.ref_dirs)

            if pop_size < len(self.ref_dirs):
                print(
                    f"WARNING: pop_size={pop_size} is less than the number of reference directions ref_dirs={len(self.ref_dirs)}.\n"
                    "This might cause unwanted behavior of the algorithm. \n"
                    "Please make sure pop_size is equal or larger than the number of reference directions. ")

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvival(ref_dirs)

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

    def _setup(self, problem, **kwargs):

        if self.ref_dirs is not None:
            if self.ref_dirs.shape[1] != problem.n_obj:
                raise Exception(
                    "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                    (self.ref_dirs.shape[1], problem.n_obj))

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            if len(self.survival.opt):
                self.opt = self.survival.opt


# =========================================================================================================
# Survival
# =========================================================================================================


class ReferenceDirectionSurvival(Survival):

    def __init__(self, ref_dirs):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs
        self.opt = None
        self.norm = HyperplaneNormalization(ref_dirs.shape[1])

    def _do(self, problem, pop, n_survive, D=None, random_state=None, **kwargs):

        # attributes to be set after the survival
        F = pop.get("F")

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # update the hyperplane based boundary estimation
        hyp_norm = self.norm
        hyp_norm.update(F, nds=non_dominated)
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(F, self.ref_dirs, ideal, nadir)

        # attributes of a population
        pop.set('rank', rank,
                'niche', niche_of_individuals,
                'dist_to_niche', dist_to_niche)

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]
        if len(self.opt) == 0:
            self.opt = pop[fronts[0]]

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(pop[last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front], random_state=random_state)

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop


@default_random_state
def niching(pop, n_remaining, niche_count, niche_of_individuals, dist_to_niche, random_state=None):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(len(pop), True)

    while len(survivors) < n_remaining:

        # number of individuals to select in this iteration
        n_select = n_remaining - len(survivors)

        # all niches where new individuals can be assigned to and the corresponding niche count
        next_niches_list = np.unique(niche_of_individuals[mask])
        next_niche_count = niche_count[next_niches_list]

        # the minimum niche count
        min_niche_count = next_niche_count.min()

        # all niches with the minimum niche count (truncate randomly if there are more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[random_state.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            random_state.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            # add the selected individual to the survivors
            mask[next_ind] = False
            survivors.append(int(next_ind))

            # increase the corresponding niche count
            niche_count[next_niche] += 1

    return survivors


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / denom
    dist_matrix = load_function("calc_perpendicular_distance")(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche, dist_matrix


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count


# =========================================================================================================
# Normalization
# =========================================================================================================


class HyperplaneNormalization:

    def __init__(self, n_dim) -> None:
        super().__init__()
        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)
        self.nadir_point = None
        self.extreme_points = None

    def update(self, F, nds=None):
        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # this decides whether only non-dominated points or all points are used to determine the extreme points
        if nds is None:
            nds = np.arange(len(F))

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[nds, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[nds, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_front, worst_of_population)


def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # add the old extreme points to never lose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points


def get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population):
    try:

        # find the intercepts using gaussian elimination
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)

        warnings.simplefilter("ignore")
        intercepts = 1 / plane

        nadir_point = ideal_point + intercepts

        # check if the hyperplane makes sense
        if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6):
            raise LinAlgError()

        # if the nadir point should be larger than any value discovered so far set it to that value
        # NOTE: different to the proposed version in the paper
        b = nadir_point > worst_point
        nadir_point[b] = worst_point[b]

    except LinAlgError:

        # fall back to worst of front otherwise
        nadir_point = worst_of_front

    # if the range is too small set it to worst of population
    b = nadir_point - ideal_point <= 1e-6
    nadir_point[b] = worst_of_population[b]

    return nadir_point


parse_doc_string(NSGA3.__init__)
