import numpy as np

from pymoo.algorithms.nsga3 import NSGA3, ReferenceDirectionSurvival
from pymoo.util.reference_direction import UniformReferenceDirectionFactory


class RNSGA3(NSGA3):

    def __init__(self,
                 ref_points,
                 pop_per_ref_point,
                 mu=0.05,
                 **kwargs):

        n_obj = ref_points.shape[1]
        n_ref_points = ref_points.shape[0]

        # add the aspiration point lines
        aspiration_ref_dirs = []
        for i in range(n_ref_points):
            ref_dirs = UniformReferenceDirectionFactory(n_dim=n_obj, n_points=pop_per_ref_point).do()
            aspiration_ref_dirs.extend(ref_dirs)
        aspiration_ref_dirs = np.array(aspiration_ref_dirs)

        kwargs['ref_dirs'] = aspiration_ref_dirs
        super().__init__(**kwargs)

        # create the survival strategy
        self.survival = AspirationPointSurvival(ref_points, aspiration_ref_dirs, mu=mu)

    def _solve(self, problem, termination):

        if self.survival.ref_points.shape[1] != problem.n_obj:
            raise Exception("Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                            (self.survival.ref_points.shape[1], problem.n_obj))

        return super()._solve(problem, termination)


class AspirationPointSurvival(ReferenceDirectionSurvival):
    def __init__(self, ref_points, aspiration_ref_dirs, mu=0.1):

        super().__init__(np.zeros((0, ref_points.shape[1])))
        self.ref_points = ref_points
        self.aspiration_ref_dirs = aspiration_ref_dirs
        self.mu = mu

    def get_ref_dirs(self):
        unit_ref_points = (self.ref_points - self.ideal_point) / (self.nadir_point - self.ideal_point)
        ref_dirs = get_ref_dirs_from_points(unit_ref_points, self.aspiration_ref_dirs, mu=self.mu)
        return ref_dirs


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
        #if not (ref_dir_for_aspiration_point > 0).min():
        #    ref_dir_for_aspiration_point[ref_dir_for_aspiration_point < 0] = 0
        #    ref_dir_for_aspiration_point = ref_dir_for_aspiration_point / np.sum(ref_dir_for_aspiration_point, axis=1)[
        #                                                                :, None]
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
