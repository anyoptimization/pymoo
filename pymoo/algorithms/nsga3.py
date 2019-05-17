import numpy as np
from numpy.linalg import LinAlgError

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.cython.function_loader import load_function
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================


class NSGA3(GeneticAlgorithm):

    def __init__(self, ref_dirs, **kwargs):
        self.ref_dirs = ref_dirs
        kwargs['individual'] = Individual(rank=np.inf, niche=-1, dist_to_niche=np.inf)
        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=30))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))
        set_if_none(kwargs, 'survival', ReferenceDirectionSurvival(ref_dirs))
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

    def _solve(self, problem, termination):
        if self.ref_dirs.shape[1] != problem.n_obj:
            raise Exception(
                "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                (self.ref_dirs.shape[1], problem.n_obj))

        return super()._solve(problem, termination)

    def _finalize(self):
        super()._finalize()
        I = np.where(self.pop.get("rank") == 0)
        pop = self.pop[I]
        if len(pop) == 1:
            self.opt = pop
        else:
            self.opt = pop[self.pop.get("is_closest")]


def comp_by_cv_then_random(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = random.choice([a, b])

    return S[:, None].astype(np.int)


class ReferenceDirectionSurvival(Survival):
    def __init__(self, ref_dirs):
        super().__init__(True)
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # attributes to be set after the survival
        F = pop.get("F")

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[non_dominated, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)

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
        niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point, self.nadir_point)
        pop.set('rank', rank, 'niche', niche_of_individuals, 'dist_to_niche',
                dist_to_niche, "is_closest", np.full(len(pop), False))

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(pop[last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop


def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    asf = np.eye(F.shape[1])
    asf[asf == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * asf[:, None, :], axis=2)
    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points


def get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population):
    try:

        # find the intercepts using gaussian elimination
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)
        intercepts = 1 / plane

        nadir_point = ideal_point + intercepts

        if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6) or np.any(nadir_point > worst_point):
            raise LinAlgError()

    except LinAlgError:
        nadir_point = worst_of_front

    b = nadir_point - ideal_point <= 1e-6
    nadir_point[b] = worst_of_population[b]

    return nadir_point


def niching(pop, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
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

        # all niches with the minimum niche count (truncate if randomly if more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[random.perm(len(next_niches))[:n_select]]

        for next_niche in next_niches:

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            next_ind = random.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
                is_closest = True
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]
                is_closest = False

            # add the selected individual to the survivors
            mask[next_ind] = False
            pop[next_ind].data["is_closest"] = is_closest
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

    return niche_of_individuals, dist_to_niche


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=np.int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count


# =========================================================================================================
# Interface
# =========================================================================================================

def nsga3(
        ref_dirs,
        pop_size=None,
        sampling=RandomSampling(),
        selection=TournamentSelection(func_comp=comp_by_cv_then_random),
        crossover=SimulatedBinaryCrossover(prob=1.0, eta=30),
        mutation=PolynomialMutation(prob=None, eta=20),
        eliminate_duplicates=True,
        n_offsprings=None,
        **kwargs):
    """

    Parameters
    ----------
    ref_dirs : {ref_dirs}
    pop_size : int (default = None)
        By default the population size is set to None which means that it will be equal to the number of reference
        line. However, if desired this can be overwritten by providing a positve number.
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    eliminate_duplicates : {eliminate_duplicates}
    n_offsprings : {n_offsprings}

    Returns
    -------
    nsga3 : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an NSGA3 algorithm object.


    """

    return NSGA3(ref_dirs,
                 pop_size=pop_size,
                 sampling=sampling,
                 selection=selection,
                 crossover=crossover,
                 mutation=mutation,
                 survival=ReferenceDirectionSurvival(ref_dirs),
                 eliminate_duplicates=eliminate_duplicates,
                 n_offsprings=n_offsprings,
                 **kwargs)


parse_doc_string(nsga3)
