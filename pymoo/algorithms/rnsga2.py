import numpy as np
from pymoo.operators.selection.random_selection import RandomSelection

from pymoo.algorithms.nsga2 import NSGA2, nsga2
from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival
from pymoo.rand import random
from pymoo.util.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================


class RankAndModifiedCrowdingSurvival(Survival):

    def __init__(self, ref_points, epsilon, weights, normalization) -> None:
        super().__init__(True)
        self.n_obj = ref_points.shape[1]
        self.ref_points = ref_points
        self.epsilon = epsilon

        self.weights = weights
        if self.weights is None:
            self.weights = np.full(self.n_obj, 1 / self.n_obj)

        self.normalization = normalization
        self.ideal_point = np.full(ref_points.shape[1], np.inf)
        self.worst_point = np.full(ref_points.shape[1], -np.inf)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F)

        if self.normalization == "ever":

            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
            self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

            self.ideal_point = np.zeros(self.n_obj)
            self.worst_point = np.ones(self.n_obj)

        elif self.normalization == "front":
            pass
        else:
            # None
            pass

        D = calc_norm_pref_distance(F, self.ref_points, self.weights, self.ideal_point, self.worst_point)

        for k, front in enumerate(fronts):

            # number of individuals remaining
            n_remaining = n_survive - len(survivors)

            # save rank and crowding in the individual class
            pop[front].set("rank", np.full(len(front), k))

            if len(front) <= n_remaining:
                pop[front].set("crowding", np.full(len(front), np.nan))
                I = np.arange(len(front))

            else:

                rank_by_distance = np.argsort(np.argsort(D[front], axis=0), axis=0)
                ref_point_of_best_rank = np.argmin(rank_by_distance, axis=1)
                ranking = rank_by_distance[np.arange(len(front)), ref_point_of_best_rank]

                # Distance from solution to every other solution
                solution_dist_matrix = calc_norm_pref_distance(F[front], F[front], self.weights, self.ideal_point,
                                                            self.worst_point)

                np.fill_diagonal(solution_dist_matrix, np.inf)

                crowding = np.full(len(front), np.nan)
                not_selected = np.arange(0, len(front))

                while len(not_selected) > 0:

                    # randomly select an alive individual
                    idx = not_selected[random.randint(0, len(not_selected))]

                    # set crowding for that individual
                    crowding[idx] = ranking[idx]

                    # need to remove myself from not-selected array
                    to_remove = [idx]

                    # Group of close solutions
                    dist = solution_dist_matrix[idx][not_selected]
                    group = not_selected[np.where(dist < self.epsilon)[0]]

                    if len(group):
                        crowding[group] = ranking[group] + np.round(len(front) / 2)

                        # remove group from not_selected array
                        to_remove.extend(group)

                    not_selected = np.array([i for i in not_selected if i not in to_remove]).astype(np.int)

                I = np.argsort(crowding)[:n_remaining]

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


def calc_norm_pref_distance(F, ref_points, weights, ideal, nadir):
    N = (((F[:, None, :] - ref_points) / (nadir - ideal)) ** 2) * weights
    N = np.sqrt(np.sum(N, axis=2))
    return N


# =========================================================================================================
# Interface
# =========================================================================================================


def rnsga2(
        ref_points,
        epsilon=0.05,
        normalization="ever",
        weights=None,
        **kwargs):
    """

    Parameters
    ----------

    Returns
    -------
    rnsga2 : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an RNSGA2 algorithm object.


    """

    rnsga2 = nsga2(**kwargs)

    rnsga2.n_obj = ref_points.shape[1]

    rnsga2.epsilon = epsilon
    rnsga2.weights = weights
    rnsga2.normalization = normalization
    rnsga2.selection = RandomSelection()
    rnsga2.survival = RankAndModifiedCrowdingSurvival(ref_points, epsilon, weights, normalization)

    return rnsga2


parse_doc_string(rnsga2)
