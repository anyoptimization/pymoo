import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import Dominator
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


# =========================================================================================================
# Implementation
# =========================================================================================================


class NSGA2(GeneticAlgorithm):

    def __init__(self, **kwargs):
        kwargs['individual'] = Individual(rank=np.inf, crowding=-1)
        super().__init__(**kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop[a].rank, b, pop[b].rank,
                               method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance_vectorized(F[front, :])
            _crowding_of_front = calc_crowding_distance_vectorized(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


def calc_crowding_distance(F):
    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, Mathematics.INF)
    else:

        # the final crowding distance result
        crowding = np.zeros(n_points)

        # for each objective
        for m in range(n_obj):

            # sort by objective randomize if they are equal
            I = np.argsort(F[:, m], kind='mergesort')
            # I = randomized_argsort(F[:, m], order='ascending')

            # norm which will be used for distance normalization
            norm = np.max(F[:, m]) - np.min(F[:, m])

            # set crowding to infinity of extreme point
            crowding[I[0]] = np.inf
            crowding[I[-1]] = np.inf

            # if norm is zero -> next objective
            if norm != 0.0:

                # add up the crowding measure for all points in between
                for i in range(1, n_points - 1):

                    # the current values to have a look at
                    _current, _last, _next = i, i - 1, i + 1

                    # if the current entry is already infinity the values will not change
                    if not np.isinf(crowding[I[_current]]):
                        crowding[I[_current]] += (F[I[_next], m] - F[I[_last], m]) / norm

    # divide by the number of objectives
    crowding = crowding / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = Mathematics.INF

    return crowding


def calc_crowding_distance_loop(F):
    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, Mathematics.INF)
    else:

        # the final crowding distance result
        crowding = np.zeros(n_points)

        # for each objective
        for m in range(n_obj):

            # sort by objective randomize if they are equal
            I = np.argsort(F[:, m], kind='mergesort')
            # I = randomized_argsort(F[:, m], order='ascending')

            # norm which will be used for distance normalization
            norm = np.max(F[:, m]) - np.min(F[:, m])

            # set crowding to infinity of extreme point
            crowding[I[0]] = np.inf
            crowding[I[-1]] = np.inf

            # if norm is zero -> next objective
            if norm != 0.0:

                # add up the crowding measure for all points in between
                for i in range(1, n_points - 1):

                    # the current values to have a look at
                    _current, _last, _next = i, i - 1, i + 1

                    # if the current entry is already infinity the values will not change
                    if not np.isinf(crowding[I[_current]]):
                        # crowding[I[_current]] += (F[I[_next], m] - F[I[_last], m]) / norm

                        # search for last and next value that are not equal
                        while _last >= 0 and F[I[_last], m] == F[I[_current], m]:
                            _last -= 1

                        while _next < n_points and F[I[_next], m] == F[I[_current], m]:
                            _next += 1

                        # if the point is in fact also an extreme point
                        if _last < 0 or _next == n_points:
                            crowding[I[_current]] = np.inf

                        # otherwise, which will be usually the case
                        else:
                            crowding[I[_current]] += (F[I[_next], m] - F[I[_last], m]) / norm

    # divide by the number of objectives
    crowding = crowding / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = Mathematics.INF

    return crowding


def calc_crowding_distance_vectorized(F, same_crowding_if_same_objective_value=False):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) \
               - np.concatenate([np.full((1, n_obj), -np.inf), F])

        dist_to_last, dist_to_next = np.copy(dist), np.copy(dist)

        if same_crowding_if_same_objective_value:
            index_dist_is_zero = np.where(dist == 0)
            for i, j in zip(*index_dist_is_zero):
                dist_to_last[i, j] = dist_to_last[i - 1, j]
            for i, j in reversed(list(zip(*index_dist_is_zero))):
                dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


# =========================================================================================================
# Interface
# =========================================================================================================


def nsga2(
        pop_size=100,
        sampling=RandomSampling(),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=None, eta=20),
        eliminate_duplicates=True,
        n_offsprings=None,
        **kwargs):
    """

    Parameters
    ----------
    pop_size : {pop_size}
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    eliminate_duplicates : {eliminate_duplicates}
    n_offsprings : {n_offsprings}

    Returns
    -------
    nsga2 : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an NSGA2 algorithm object.


    """

    return NSGA2(pop_size=pop_size,
                 sampling=sampling,
                 selection=selection,
                 crossover=crossover,
                 mutation=mutation,
                 survival=RankAndCrowdingSurvival(),
                 eliminate_duplicates=eliminate_duplicates,
                 n_offsprings=n_offsprings,
                 **kwargs)


parse_doc_string(nsga2)
