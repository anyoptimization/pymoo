import numpy as np
from scipy import special

from pymoo.util.plotting import plot_3d





def get_ref_dirs_from_section(n_obj, n_sections):
    if n_obj == 1:
        return np.array([1.0])

    # all possible values for the vector
    sections = np.linspace(0, 1, num=n_sections + 1)[::-1]

    M = []
    ref_recursive([], sections, 0, n_obj, M)
    M = np.array(M)
    M[M == 0] = 0.000001

    return M


def get_multi_layer_ref_dirs(n_obj, p_and_scaling):
    ref_dirs = []

    for p, scaling in p_and_scaling:
        r = get_ref_dirs_from_section(n_obj, p)
        r = r * scaling + ((1 - scaling) / n_obj)
        ref_dirs.append(r)
    ref_dirs = np.concatenate(ref_dirs, axis=0)
    return ref_dirs


# returns the closest possible number of references lines to given one
def get_ref_dirs_from_n(n_obj, n_refs, max_sections=1000):

    if n_obj == 1:
        return np.array([[1.0]])

    n_sections = np.array([get_number_of_reference_directions(n_obj, i) for i in range(max_sections)])
    idx = np.argmin((n_sections <= n_refs).astype(np.int))
    M = get_ref_dirs_from_section(n_obj, idx - 1)
    return M


def ref_recursive(v, sections, level, max_level, result):
    v_sum = np.sum(np.array(v))

    # sum slightly above or below because of numerical issues
    if v_sum > 1.0001:
        return
    elif level == max_level:
        if 1.0 - v_sum < 0.0001:
            result.append(v)
    else:
        for e in sections:
            next = list(v)
            next.append(e)
            ref_recursive(next, sections, level + 1, max_level, result)


def get_number_of_reference_directions(n_obj, n_sections):
    return int(special.binom(n_obj + n_sections - 1, n_sections))


# returns the closest possible number of references lines to given one
def get_uniform_weights(n_points, n_dim, max_sections=300):
    def n_uniform_weights(n_obj, n_sections):
        return int(special.binom(n_obj + n_sections - 1, n_sections))

    # Generates uniform distribution of reference points
    n_sections = np.array([n_uniform_weights(n_dim, i) for i in range(max_sections)])
    idx = np.argmin((n_sections <= n_points).astype(np.int))
    M = get_ref_dirs_from_section(n_dim, idx - 1)

    return M


def get_ref_dirs_from_points(ref_point, ref_dirs, mu=0.1, method='uniform', p=None):
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


if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt

    ref_dirs = get_multi_layer_ref_dirs(3, [(12, 1.0), (12, 0.1)])

    plot_3d(ref_dirs)
    exit(0)

    pt = np.array([[0.4, 0.6, 0.2], [0.8, 0.6, 0.8]])
    # pt = np.array([[0.1, 2, 1]])
    start = time.time()
    w = get_ref_dirs_from_points(pt, n_obj=3, pop_size=100)
    print(f'{time.time()-start}')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w[:, 0], w[:, 1], w[:, 2])
    ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2])
    for p in pt:
        p = p - np.dot(p - np.array([1, 0, 0]), np.ones(3) / np.sqrt(3)) * np.ones(3) / np.sqrt(3)
        p_p = p
        p_p[p < 0] = 0
        ax.scatter(p[0], p[1], p[2], c='r', marker='d')
        ax.scatter(p_p[0], p_p[1], p_p[2], c='r', marker='*')
    plt.show()
