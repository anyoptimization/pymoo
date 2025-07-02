import numpy as np

from pymoo.algorithms.moo.nsga3 import calc_niche_count, niching, comp_by_cv_then_random, associate_to_niches, NSGA3
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.misc import intersect
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import denormalize, get_extreme_points_c, get_nadir_point
from pymoo.util.reference_direction import UniformReferenceDirectionFactory


# =========================================================================================================
# Implementation
# =========================================================================================================


class RNSGA3(NSGA3):

    def __init__(self,
                 ref_points,
                 pop_per_ref_point,
                 mu=0.05,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 **kwargs):
        """

        Parameters
        ----------

        ref_points : {ref_points}
        
        pop_per_ref_point : int
            Size of the population used for each reference point.

        mu : float
            Defines the init_simplex_scale of the reference lines used during survival selection. Increasing mu will result
            having solutions with a larger spread.

        n_offsprings : {n_offsprings}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}

        """

        # number of objectives the reference lines have
        n_obj = ref_points.shape[1]

        # add the aspiration point lines
        aspiration_ref_dirs = UniformReferenceDirectionFactory(n_dim=n_obj, n_points=pop_per_ref_point).do()

        survival = AspirationPointSurvival(ref_points, aspiration_ref_dirs, mu=mu)
        pop_size = ref_points.shape[0] * aspiration_ref_dirs.shape[0] + aspiration_ref_dirs.shape[1]
        ref_dirs = None

        super().__init__(ref_dirs,
                         pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         **kwargs)

    def _setup(self, problem, **kwargs):
        if self.survival.ref_points.shape[1] != problem.n_obj:
            raise Exception("Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                            (self.survival.ref_points.shape[1], problem.n_obj))

    def _finalize(self):
        pass


class AspirationPointSurvival(Survival):

    def __init__(self, ref_points, aspiration_ref_dirs, mu=0.1):
        super().__init__()

        self.ref_points = ref_points
        self.aspiration_ref_dirs = aspiration_ref_dirs
        self.mu = mu

        self.ref_dirs = aspiration_ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.opt = None
        self.ideal_point = np.full(ref_points.shape[1], np.inf)
        self.worst_point = np.full(ref_points.shape[1], -np.inf)

    def _do(self, problem, pop, n_survive, D=None, random_state=None, **kwargs):

        # attributes to be set after the survival
        F = pop.get("F")

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F, self.ref_points)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F, self.ref_points)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(
            np.vstack([F[non_dominated], self.ref_points])
            , self.ideal_point,
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

        unit_ref_points = (self.ref_points - self.ideal_point) / (self.nadir_point - self.ideal_point)
        ref_dirs = get_ref_dirs_from_points(unit_ref_points, self.aspiration_ref_dirs, mu=self.mu)
        self.ref_dirs = denormalize(ref_dirs, self.ideal_point, self.nadir_point)

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(F, ref_dirs, self.ideal_point,
                                                                               self.nadir_point)
        pop.set('rank', rank, 'niche', niche_of_individuals, 'dist_to_niche', dist_to_niche)

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(pop[last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front], random_state=random_state)

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop


def get_ref_dirs_from_points(ref_point, ref_dirs, mu=0.1):
    """
    This function takes user specified reference points, and creates smaller sets of equidistant
    Das-Dennis points around the projection of user points on the Das-Dennis hyperplane
    :param ref_point: List of user specified reference points
    :param n_obj: Number of objectives to consider
    :param mu: Shrinkage factor (0-1), Smaller = tighter convergence, Larger= larger convergence
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
        #  > 1.0: in front of p1.
        w = p0 - l0
        d = np.dot(w, p_no) / dot
        l = l * d
        return l0 + l
    else:
        # The segment is parallel to plane then return the perpendicular projection
        ref_proj = l1 - (np.dot(l1 - p0, p_no) * p_no)
        return ref_proj


parse_doc_string(RNSGA3.__init__)
