import copy
import numpy as np
from scipy import special


def get_ref_dirs_from_section(n_obj, n_sections):

    if n_obj == 1:
        return np.array([1.0])

    # all possible values for the vector
    sections = np.linspace(0, 1, num=n_sections + 1)[::-1]

    ref_dirs = []
    ref_recursive([], sections, 0, n_obj, ref_dirs)
    return np.array(ref_dirs)


# returns the closest possible number of references lines to given one
def get_ref_dirs_from_n(n_obj, n_refs, max_sections=100):
    n_sections = np.array([get_number_of_reference_directions(n_obj, i) for i in range(max_sections)])
    idx = np.argmin((n_sections < n_refs).astype(np.int))
    M = get_ref_dirs_from_section(n_obj, idx-1)
    M[M==0] = 0.000001
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


def get_ref_dirs_from_points(points, n_obj):

    extreme_reference_points = np.eye(n_obj)

    ref_points = np.append(points, extreme_reference_points, axis=0)
    return ref_points

def get_ref_dirs_from_points(ref_point, n_obj, pop_size, alpha=0.1, method='uniform'):
    """
    This function takes user specified reference points, and creates smaller sets of equidistant
    Das-Dennis points around the projection of user points on the Das-Dennis hyperplane
    :param ref_point: List of user specified reference points
    :param n_obj: Number of objectives to consider
    :param alpha: Shrinkage factor (0-1), Smaller = tigher convergence, Larger= larger convergence
    :return: Set of reference points
    """
    ref_dirs = []
    n_vector = np.ones(n_obj) / np.linalg.norm(np.ones(n_obj))  # Normal vector of Das Dennis plane
    point_on_plane = np.eye(n_obj)[0]  # Point on Das-Dennis
    # point_on_plane = np.array([0.333333]*n_obj)
    pop_size = pop_size/len(ref_point)  # Limit the number of Das-Dennis points
    if method == 'uniform':
        reference_directions = get_ref_dirs_from_n(n_obj, pop_size)  # Das-Dennis points
    elif method == 'random':
        reference_directions = np.random.rand(int(pop_size), int(n_obj))
        reference_directions = reference_directions - np.dot(reference_directions - point_on_plane, n_vector)[:, None] * n_vector
    elif method =='nested':
        reference_directions = get_ref_dirs_from_section(n_obj, 3)  # Das-Dennis points
        # ref_dirs.extend(reference_directions)
        nested = get_ref_dirs_from_section(n_obj, 2)
        nested = 0.3333333 + 0.5 * (nested - 0.3333333) # Shrink the nested region
        nested = (nested - np.dot(nested - point_on_plane, n_vector)[:, None] * n_vector)
        reference_directions = np.vstack((reference_directions, nested))
        # return np.array(ref_dirs)

    for point in ref_point:
        ref_proj = point - np.dot(point - point_on_plane, n_vector) * n_vector
        # TODO: Compute which is faster, a copy.deepcopy, or recomputing all the points from get_ref_dirs_from_n
        ref_dir = copy.deepcopy(reference_directions)  # Copy of computed reference directions
        for i in range(n_obj):  # Shrink Das-Dennis points by a factor of alpha
            # ref_dir[:, i] = point[i] + alpha * (ref_dir[:, i] - point[i])
            ref_dir[:, i] = ref_proj[i] + alpha * (ref_dir[:, i] - ref_proj[i])
        for d in ref_dir:  # Project shrunked Das-Dennis points back onto original Das-Dennis hyperplane
            ref_dirs.append(d - np.dot(d - point_on_plane, n_vector) * n_vector)
    # TODO: Extreme points are only extreme of the scale is normalized between 0-1, how to make them truly extreme?
    ref_dirs.extend(np.eye(n_obj))  # Add extreme points
    # extreme = get_ref_dirs_from_n(n_obj, 24)
    # ref_dirs.extend(extreme)
    return np.array(ref_dirs)

def norm(F):
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)


if __name__ == '__main__':

    w = get_ref_dirs_from_points(np.array([[0.2, 0.6, 0.8]]), n_obj=3, pop_size=100, method='nested')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w[:, 0], w[:, 1], w[:, 2])
    plt.show()
