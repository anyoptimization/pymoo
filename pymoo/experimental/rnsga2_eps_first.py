import numpy as np

from pymoo.algorithms.nsga2 import nsga2
from pymoo.algorithms.nsga3 import get_extreme_points_c
from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================

class RankAndModifiedCrowdingSurvival(Survival):

    def __init__(self, ref_points,
                 epsilon,
                 weights,
                 normalization,
                 survival_type,
                 extreme_points_as_reference_points
                 ) -> None:

        super().__init__(True)
        self.n_obj = ref_points.shape[1]
        self.ref_points = ref_points
        self.epsilon = epsilon
        self.survival_type = survival_type
        self.extreme_points_as_reference_points = extreme_points_as_reference_points

        self.weights = weights
        if self.weights is None:
            self.weights = np.full(self.n_obj, 1 / self.n_obj)

        self.normalization = normalization
        self.ideal_point = np.full(self.n_obj, np.inf)
        self.nadir_point = np.full(self.n_obj, -np.inf)

    def _do(self, pop, n_survive, **kwargs):

        # get the objective space values and objects
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts, rank = NonDominatedSorting().do(F, return_rank=True)

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

        # matrix that contains distance to other solutions - set to infinity if not in the same front
        dist_to_others = np.full((len(pop), len(pop)), np.inf)

        # assign solutions to the corresponding reference points
        assigned_to_ref_point = np.full((len(pop), len(self.ref_points)), False)
        n_assinged_to_ref_point = np.zeros(len(self.ref_points))

        # now the rank and crowding front wise
        for k, front in enumerate(fronts):

            # save rank attributes to the individuals - rank = front here
            pop[front].set("rank", np.full(len(front), k))

            # boolean mask over the points that were considered
            not_selected = [e for e in front]
            while len(not_selected) > 0:

                # assign each solution to a reference point (try to assign as uniformly as possible)
                for k in np.argsort(n_assinged_to_ref_point)[:len(not_selected)]:
                    n_assinged_to_ref_point[k] += 1
                    closest_to_ref_point = not_selected[np.argmin(dist_to_ref_points[not_selected, k])]

                    assigned_to_ref_point[closest_to_ref_point, k] = True
                    not_selected = [e for e in not_selected if e != closest_to_ref_point]

            # distance from solution to every other solution and set distance to itself to infinity
            dist_in_front = calc_norm_pref_distance(F[front], F[front], self.weights, self.ideal_point,
                                                    self.nadir_point)

            for i, a in enumerate(front):
                for j, b in enumerate(front):
                    #if np.all(assigned_to_ref_point[a] == assigned_to_ref_point[b]):
                    dist_to_others[a, b] = dist_in_front[i, j]

        np.fill_diagonal(dist_to_others, np.inf)

        # all individuals that are currently not surviving
        not_surviving = np.full(len(pop), True)

        # all individuals that are cleared because they were too close to already selected solutions
        not_cleared = np.full(len(pop), True)

        # this is a counter for the survival circle (used to tournament selection later)
        counter = 0

        # until not all points were selected
        while len(survivors) < n_survive:

            # iterate over all reference points
            for k in range(len(self.ref_points)):

                # all solutions to be still considered
                b = np.where(np.logical_and(np.logical_and(not_surviving, not_cleared), assigned_to_ref_point[:, k]))[0]

                # restart the clearing because no solutions found to consider
                if len(b) == 0:
                    not_cleared[np.arange(len(pop))] = True
                    b = np.where(np.logical_and(not_surviving, assigned_to_ref_point[:, k]))[0]

                # no prefer non-dominated solutions first
                b = b[np.where(rank[b] == rank[b].min())]

                # find the solution which is closest and not selected yet
                I = b[np.argmin(dist_to_ref_points[b, k])]

                # for all solutions that are too close
                for i in np.where(dist_to_others[I] < self.epsilon)[0]:
                    not_cleared[i] = False

                survivors.append(I)
                not_surviving[I] = False
                pop[I].data["crowding"] = - counter

            counter += 1

        return pop[survivors]


def calc_norm_pref_distance(A, B, weights, ideal, nadir):
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    N = ((D / (nadir - ideal)) ** 2) * weights
    N = np.sqrt(np.sum(N, axis=1) * len(weights))
    return np.reshape(N, (A.shape[0], B.shape[0]))


# =========================================================================================================
# Interface
# =========================================================================================================


def rnsga2(
        ref_points,
        epsilon=0.001,
        normalization="front",
        weights=None,
        survival_type="closest",
        extreme_points_as_reference_points=False,
        **kwargs):
    """


    Parameters
    ----------

    ref_points : {ref_points}

    epsilon : float

    weights : np.array

    normalization : {{'no', 'front', 'ever'}}

    survival_type : {{'closest', 'random'}}

    extreme_points_as_reference_points : bool



    Returns
    -------
    rnsga2 : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an RNSGA2 algorithm object.


    """

    rnsga2 = nsga2(**kwargs)

    rnsga2.epsilon = epsilon
    rnsga2.weights = weights
    rnsga2.normalization = normalization
    rnsga2.selection = RandomSelection()
    rnsga2.survival = RankAndModifiedCrowdingSurvival(ref_points, epsilon, weights, normalization,
                                                      survival_type, extreme_points_as_reference_points)

    return rnsga2


parse_doc_string(rnsga2)
