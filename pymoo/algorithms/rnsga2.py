import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import get_extreme_points_c
from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================

class RNSGA2(NSGA2):

    def __init__(self,
                 ref_points,
                 epsilon=0.001,
                 normalization="front",
                 weights=None,
                 extreme_points_as_reference_points=False,
                 **kwargs):
        """

        Parameters
        ----------

        ref_points : {ref_points}
        epsilon : float
        weights : np.array
        normalization : {{'no', 'front', 'ever'}}
        extreme_points_as_reference_points : bool

        """

        self.epsilon = epsilon
        self.weights = weights
        self.normalization = normalization
        self.selection = RandomSelection()

        super().__init__(**kwargs)

        self.survival = RankAndModifiedCrowdingSurvival(ref_points, epsilon, weights, normalization,
                                                        extreme_points_as_reference_points)


class RankAndModifiedCrowdingSurvival(Survival):

    def __init__(self, ref_points,
                 epsilon,
                 weights,
                 normalization,
                 extreme_points_as_reference_points
                 ) -> None:

        super().__init__(True)
        self.n_obj = ref_points.shape[1]
        self.ref_points = ref_points
        self.epsilon = epsilon
        self.extreme_points_as_reference_points = extreme_points_as_reference_points

        self.weights = weights
        if self.weights is None:
            self.weights = np.full(self.n_obj, 1 / self.n_obj)

        self.normalization = normalization
        self.ideal_point = np.full(self.n_obj, np.inf)
        self.nadir_point = np.full(self.n_obj, -np.inf)

    def _do(self, problem, pop, n_survive, **kwargs):

        # get the objective space values and objects
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F)

        if self.normalization == "ever":
            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
            self.nadir_point = np.max(np.vstack((self.nadir_point, F)), axis=0)

        elif self.normalization == "front":
            front = fronts[0]
            if len(front) > 1:
                self.ideal_point = np.min(F[front], axis=0)
                self.nadir_point = np.max(F[front], axis=0)

        elif self.normalization == "no":
            self.ideal_point = np.zeros(self.n_obj)
            self.nadir_point = np.ones(self.n_obj)

        if self.extreme_points_as_reference_points:
            self.ref_points = np.row_stack([self.ref_points, get_extreme_points_c(F, self.ideal_point)])

        # calculate the distance matrix from ever solution to all reference point
        dist_to_ref_points = calc_norm_pref_distance(F, self.ref_points, self.weights, self.ideal_point,
                                                     self.nadir_point)

        for k, front in enumerate(fronts):

            # save rank attributes to the individuals - rank = front here
            pop[front].set("rank", np.full(len(front), k))

            # number of individuals remaining
            n_remaining = n_survive - len(survivors)

            # the ranking of each point regarding each reference point (two times argsort is necessary)
            rank_by_distance = np.argsort(np.argsort(dist_to_ref_points[front], axis=0), axis=0)

            # the reference point where the best ranking is coming from
            ref_point_of_best_rank = np.argmin(rank_by_distance, axis=1)

            # the actual ranking which is used as crowding
            ranking = rank_by_distance[np.arange(len(front)), ref_point_of_best_rank]

            if len(front) <= n_remaining:

                # we can simply copy the crowding to ranking. not epsilon selection here
                crowding = ranking
                I = np.arange(len(front))

            else:

                # Distance from solution to every other solution and set distance to itself to infinity
                dist_to_others = calc_norm_pref_distance(F[front], F[front], self.weights, self.ideal_point,
                                                         self.nadir_point)
                np.fill_diagonal(dist_to_others, np.inf)

                # the crowding that will be used for selection
                crowding = np.full(len(front), np.nan)

                # solutions which are not already selected - for
                not_selected = np.argsort(ranking)

                # until we have saved a crowding for each solution
                while len(not_selected) > 0:

                    # select the closest solution
                    idx = not_selected[0]

                    # set crowding for that individual
                    crowding[idx] = ranking[idx]

                    # need to remove myself from not-selected array
                    to_remove = [idx]

                    # Group of close solutions
                    dist = dist_to_others[idx][not_selected]
                    group = not_selected[np.where(dist < self.epsilon)[0]]

                    # if there exists solution with a distance less than epsilon
                    if len(group):
                        # discourage them by giving them a high crowding
                        crowding[group] = ranking[group] + np.round(len(front) / 2)

                        # remove group from not_selected array
                        to_remove.extend(group)

                    not_selected = np.array([i for i in not_selected if i not in to_remove])

                # now sort by the crowding (actually modified rank) ascending and let the best survive
                I = np.argsort(crowding)[:n_remaining]

            # set the crowding to all individuals
            pop[front].set("crowding", crowding)

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        # inverse of crowding because nsga2 does maximize it (then tournament selection can stay the same)
        pop.set("crowding", -pop.get("crowding"))

        return pop[survivors]


def calc_norm_pref_distance(A, B, weights, ideal, nadir):
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    N = ((D / (nadir - ideal)) ** 2) * weights
    N = np.sqrt(np.sum(N, axis=1) * len(weights))
    return np.reshape(N, (A.shape[0], B.shape[0]))


parse_doc_string(RNSGA2.__init__)
