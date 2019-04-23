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

        # always count how many are assigned to each point
        n_assigned_to_ref_point = np.zeros(len(self.ref_points))

        # then front-wise and domination first
        for k, front in enumerate(fronts):

            # save rank attributes to the individuals - rank = front here
            pop[front].set("rank", np.full(len(front), k))

            # number of individuals remaining
            n_remaining = n_survive - len(survivors)

            # assign each solution to a reference point
            assigned_to_ref_point = np.argmin(dist_to_ref_points[front], axis=1)

            # if we can add the front without any split
            if len(front) <= n_remaining:

                # the whole front survives
                I = np.arange(len(front))

                # there is not other criterium here to decide which solution is better
                crowding = np.full(len(front), 0)

                # count the assignments to the reference points
                ref_point, count = np.unique(assigned_to_ref_point, return_counts=True)
                n_assigned_to_ref_point[ref_point] += count

            else:
                n_remaining = n_survive - len(survivors)

                # the  crowding vector to be finally returned
                crowding = np.full(len(front), 0)

                # then always take points from underrepresented reference points first
                not_selected = np.full(len(front), True)

                # all individuals that are cleared because they were too close to already I solutions
                not_cleared = np.full(len(front), True)

                # Distance from solution to every other solution and set distance to itself to infinity
                dist_to_others = calc_norm_pref_distance(F[front], F[front], self.weights, self.ideal_point,
                                                         self.nadir_point)
                np.fill_diagonal(dist_to_others, np.inf)

                while n_remaining > 0:

                    # first select from underrepresented reference points
                    ref_point = np.argmin(n_assigned_to_ref_point)

                    # now add the clearing to create the selection mask
                    b = np.where(np.logical_and(not_selected, not_cleared))[0]

                    # when clearing caused that no solutions are there to select - remove clearing
                    if len(b) == 0:
                        not_cleared[:] = True
                        b = np.where(not_selected)[0]

                    # now select the solution that is the closest to reference point
                    I = b[np.argmin(dist_to_ref_points[front[b], ref_point])]

                    # for all solutions that are too close to the selected one
                    too_close = np.where(dist_to_others[I] < self.epsilon)[0]
                    not_cleared[too_close] = False

                    crowding[I] = n_assigned_to_ref_point[ref_point]
                    n_assigned_to_ref_point[ref_point] += 1

                    n_remaining -= 1
                    not_selected[I] = False

                I = np.where(np.logical_not(not_selected))[0]

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
