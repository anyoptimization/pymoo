import numpy as np

from pymoo.model.survival import Survival, split_by_feasibility
from pymoo.operators.survival.reference_line_survival import associate_to_niches, calc_niche_count, niching, \
    get_extreme_points, get_intercepts
from pymoo.util.mathematics import Mathematics
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class AspirationPointSurvival(Survival):
    def __init__(self, ref_points, aspiration_ref_dirs, mu=0.1):
        super().__init__()

        self.aspiration_ref_dirs = aspiration_ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.ideal_point = np.full(aspiration_ref_dirs.shape[1], np.inf)

        self.ref_points = ref_points
        self.mu = mu

    def _do(self, pop, n_survive, D=None, **kwargs):

        # convert to integer for later usage
        n_survive = int(n_survive)

        # first split by feasibility for normalization
        feasible, infeasible = split_by_feasibility(pop)

        # number of survivors from the feasible population 
        # in case of having not enough feasible solution all feasible will survive
        if len(feasible) < n_survive:
            n_survive_feasible = len(feasible)
        else:
            n_survive_feasible = n_survive

        # attributes to be set after the survival
        survivors, rank, niche_of_individuals, dist_to_niche = [], [], [], []

        # if there are feasible solutions to survive
        if len(feasible) > 0:

            # consider only feasible solutions form now on
            F = pop.F[feasible, :]

            # find or usually update the new ideal point - from feasible solutions
            self.ideal_point = np.min(np.concatenate([self.ideal_point[None, :], F], axis=0), axis=0)

            # calculate the fronts of the population
            fronts, _rank = NonDominatedSorting(epsilon=Mathematics.EPS).do(F, return_rank=True,
                                                                            n_stop_if_ranked=n_survive_feasible)
            non_dominated, last_front = fronts[0], fronts[-1]

            # calculate the worst point of feasible individuals
            worst_point = np.max(np.vstack((F, self.ref_points)), axis=0)
            # calculate the nadir point from non dominated individuals
            nadir_point = np.max(np.vstack((F[non_dominated, :], self.ref_points)), axis=0)

            # find the extreme points for normalization
            self.extreme_points = get_extreme_points(np.vstack((F, self.ref_points)), self.ideal_point,
                                                     extreme_points=self.extreme_points)

            # find the intercepts for normalization and do backup if gaussian elimination fails
            self.intercepts = get_intercepts(self.extreme_points, self.ideal_point, nadir_point, worst_point)

            unit_ref_points = (self.ref_points - self.ideal_point) / self.intercepts
            ref_dirs = get_ref_dirs_from_points(unit_ref_points, self.aspiration_ref_dirs, mu=self.mu)

            # index of the first n fronts form now on - including splitting front
            I = np.concatenate(fronts)
            F = F[I, :]

            # associate individuals to niches
            niche_of_individuals, dist_to_niche = associate_to_niches(F, ref_dirs, self.ideal_point,
                                                                      self.intercepts)

            # if a splitting of the last front is not necessary
            if F.shape[0] == n_survive_feasible:
                _survivors = np.arange(F.shape[0])

            # otherwise we have to select using niching
            else:

                _last_front = np.arange(len(I) - len(last_front), len(I))
                _until_last_front = np.arange(0, len(I) - len(last_front))

                _survivors = []
                n_remaining = n_survive_feasible
                niche_count = np.zeros(len(ref_dirs), dtype=np.int)

                if len(fronts) > 1:
                    _survivors.extend(_until_last_front)
                    niche_count = calc_niche_count(len(ref_dirs),
                                                   niche_of_individuals[_until_last_front])
                    n_remaining -= len(_until_last_front)

                S = niching(F[_last_front, :], n_remaining, niche_count, niche_of_individuals[_last_front],
                            dist_to_niche[_last_front])

                # S = niching_vectorized(F[_last_front, :], n_remaining, niche_count, niche_of_individuals[_last_front],
                #                       dist_to_niche[_last_front])

                _survivors.extend(_last_front[S].tolist())

            # reindex the survivors to the absolute index
            survivors = feasible[I[_survivors]]

            # save the attributes for surviving individuals
            rank = _rank[I[_survivors]]
            niche_of_individuals = niche_of_individuals[_survivors]
            dist_to_niche = dist_to_niche[_survivors]

        # if we need to fill up with infeasible solutions - we do so. Also, the data structured need to be reindexed
        n_infeasible = n_survive - len(survivors)
        if n_infeasible > 0:
            survivors = np.concatenate([survivors, infeasible[:n_infeasible]])
            rank = np.concatenate([rank, Mathematics.INF * np.ones(n_infeasible)])
            niche_of_individuals = np.concatenate([niche_of_individuals, -1 * np.ones(n_infeasible)])
            dist_to_niche = np.concatenate([dist_to_niche, Mathematics.INF * np.ones(n_infeasible)])

        # set attributes globally for other modules
        if D is not None:
            D['rank'] = rank
            D['niche'] = niche_of_individuals
            D['dist_to_niche'] = dist_to_niche

        # now truncate the population
        pop.filter(survivors)


def get_ref_dirs_from_points(ref_point, ref_dirs, mu=0.1):
    """
    This function takes user specified reference points, and creates smaller sets of equidistant
    Das-Dennis points around the projection of user points on the Das-Dennis hyperplane
    :param ref_point: List of user specified reference points
    :param n_obj: Number of objectives to consider
    :param mu: Shrinkage factor (0-1), Smaller = tigher convergence, Larger= larger convergence
    :return: Set of reference points
    """

    n_obj = ref_point.shape[1]

    val = []
    n_vector = np.ones(n_obj) / np.sqrt(n_obj)  # Normal vector of Das Dennis plane
    point_on_plane = np.eye(n_obj)[0]  # Point on Das-Dennis

    for point in ref_point:

        ref_dir_for_aspiration_point = np.copy(ref_dirs)  # Copy of computed reference directions
        ref_dir_for_aspiration_point = mu * ref_dir_for_aspiration_point

        cent = np.mean(ref_dir_for_aspiration_point, axis=0)  # Find centroid of shrunken reference points

        # Project shrunken Das-Dennis points back onto original Das-Dennis hyperplane
        intercept = line_plane_intersection(np.zeros(n_obj), point, point_on_plane, n_vector)
        shift = intercept - cent  # shift vector

        ref_dir_for_aspiration_point += shift

        # If reference directions are located outside of first octant, redefine points onto the border
        if not (ref_dir_for_aspiration_point > 0).min():
            ref_dir_for_aspiration_point[ref_dir_for_aspiration_point < 0] = 0
            ref_dir_for_aspiration_point = ref_dir_for_aspiration_point / np.sum(ref_dir_for_aspiration_point, axis=1)[
                                                                          :, None]
        val.extend(ref_dir_for_aspiration_point)

    val.extend(np.eye(n_obj))  # Add extreme points
    return np.array(val)


# intersection function

def line_plane_intersection(l0, l1, p0, p_no, epsilon=1e-6):
    """
    l0, l1: define the line
    p0, p_no: define the plane:
        p0 is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction;
             (does not need to be normalized).

    reference: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    return a Vector or None (when the intersection can't be found).
    """

    l = l1 - l0
    dot = np.dot(l, p_no)

    if abs(dot) > epsilon:
        # the factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = p0 - l0
        d = np.dot(w, p_no) / dot
        l = l * d
        return l0 + l
    else:
        # The segment is parallel to plane then return the perpendicular projection
        ref_proj = l1 - (np.dot(l1 - p0, p_no) * p_no)
        return ref_proj
