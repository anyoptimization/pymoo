import sys

import numpy as np
from scipy import special

from pymoo.util.misc import find_duplicates, cdist
from pymoo.util import default_random_state


# =========================================================================================================
# Model
# =========================================================================================================


def default_ref_dirs(m):
    if m == 1:
        return np.array([[1.0]])
    elif m == 2:
        return UniformReferenceDirectionFactory(m, n_partitions=99).do()
    elif m == 3:
        return UniformReferenceDirectionFactory(m, n_partitions=12).do()
    else:
        raise Exception("No default reference directions for more than 3 objectives. Please provide them directly:"
                        "https://pymoo.org/misc/reference_directions.html")


class ReferenceDirectionFactory:

    def __init__(self, n_dim, scaling=None, lexsort=True, verbose=False, **kwargs) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.scaling = scaling
        self.lexsort = lexsort
        self.verbose = verbose

    def __call__(self):
        return self.do()

    @default_random_state(seed=1)
    def do(self, random_state=None):

        if self.n_dim == 1:
            return np.array([[1.0]])
        else:

            val = self._do(random_state=random_state)
            if isinstance(val, tuple):
                ref_dirs, other = val[0], val[1:]
            else:
                ref_dirs = val

            if self.scaling is not None:
                ref_dirs = scale_reference_directions(ref_dirs, self.scaling)

            # do ref_dirs is desired
            if self.lexsort:
                I = np.lexsort([ref_dirs[:, j] for j in range(ref_dirs.shape[1])][::-1])
                ref_dirs = ref_dirs[I]

            return ref_dirs

    def _do(self, random_state=None):
        return None


# =========================================================================================================
# Das Dennis Reference Directions (Uniform)
# =========================================================================================================


def get_number_of_uniform_points(n_partitions, n_dim):
    """
    Returns the number of uniform points that can be created uniformly.
    """
    return int(special.binom(n_dim + n_partitions - 1, n_partitions))


def get_partition_closest_to_points(n_points, n_dim):
    """
    Returns the corresponding partition number which create the desired number of points
    or less!
    """

    if n_dim == 1:
        return 0

    n_partitions = 1
    _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    while _n_points <= n_points:
        n_partitions += 1
        _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    return n_partitions - 1


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


class UniformReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim, scaling=None, n_points=None, n_partitions=None, **kwargs) -> None:
        super().__init__(n_dim, scaling=scaling, **kwargs)

        if n_points is not None:
            n_partitions = get_partition_closest_to_points(n_points, n_dim)
            results_in = get_number_of_uniform_points(n_partitions, n_dim)

            # the number of points are not matching to any partition number
            if results_in != n_points:
                results_in_next = get_number_of_uniform_points(n_partitions + 1, n_dim)
                raise Exception("The number of points (n_points = %s) can not be created uniformly.\n"
                                "Either choose n_points = %s (n_partitions = %s) or "
                                "n_points = %s (n_partitions = %s)." %
                                (n_points, results_in, n_partitions, results_in_next, n_partitions + 1))

            self.n_partitions = n_partitions

        elif n_partitions is not None:
            self.n_partitions = n_partitions

        else:
            raise Exception("Either provide number of partitions or number of points.")

    def _do(self, random_state=None):
        return das_dennis(self.n_partitions, self.n_dim)


# =========================================================================================================
# Multi Layer
# =========================================================================================================


class MultiLayerReferenceDirectionFactory:

    def __init__(self, *args) -> None:
        self.layers = []
        self.layers.extend(args)

    def __call__(self):
        return self.do()

    def add_layer(self, *args):
        self.layers.extend(args)

    def do(self):
        ref_dirs = []
        for factory in self.layers:
            ref_dirs.append(factory)
        ref_dirs = np.concatenate(ref_dirs, axis=0)
        is_duplicate = find_duplicates(ref_dirs)
        return ref_dirs[np.logical_not(is_duplicate)]


# =========================================================================================================
# Util
# =========================================================================================================

@default_random_state
def get_rng(random_state=None, **kwargs):
    return random_state


@default_random_state
def sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="kraemer", random_state=None, **kwargs):
    if unit_simplex_mapping == "sum":
        rnd = map_onto_unit_simplex(random_state.random((n_points, n_dim)), "sum")

    elif unit_simplex_mapping == "kraemer":
        rnd = map_onto_unit_simplex(random_state.random((n_points, n_dim)), "kraemer")

    elif unit_simplex_mapping == "das-dennis":
        n_partitions = get_partition_closest_to_points(n_points, n_dim)
        rnd = UniformReferenceDirectionFactory(n_dim, n_partitions=n_partitions).do()

    else:
        raise Exception("Please define a valid sampling on unit simplex strategy!")

    return rnd


def map_onto_unit_simplex(rnd, method):
    n_points, n_dim = rnd.shape

    if method == "sum":
        ret = rnd / rnd.sum(axis=1)[:, None]

    elif method == "kraemer":
        M = sys.maxsize

        rnd *= M
        rnd = rnd[:, :n_dim - 1]
        rnd = np.column_stack([np.zeros(n_points), rnd, np.full(n_points, M)])

        rnd = np.sort(rnd, axis=1)

        ret = np.full((n_points, n_dim), np.nan)
        for i in range(1, n_dim + 1):
            ret[:, i - 1] = rnd[:, i] - rnd[:, i - 1]
        ret /= M

    else:
        raise Exception("Invalid unit simplex mapping!")

    return ret


def scale_reference_directions(ref_dirs, scaling):
    return ref_dirs * scaling + ((1 - scaling) / ref_dirs.shape[1])


def select_points_with_maximum_distance(X, n_select, selected=[], random_state=None):
    n_points, n_dim = X.shape

    # calculate the distance matrix
    D = cdist(X, X)

    # if no selection provided pick randomly in the beginning
    if len(selected) == 0:
        # random_state should be provided by caller
        selected = [random_state.integers(len(X))]

    # create variables to store what selected and what not
    not_selected = [i for i in range(n_points) if i not in selected]

    # remove unnecessary points
    dist_to_closest_selected = D[:, selected].min(axis=1)

    # now select the points until sufficient ones are found
    while len(selected) < n_select:
        # find point that has the maximum distance to all others
        index_in_not_selected = dist_to_closest_selected[not_selected].argmax()
        I = not_selected[index_in_not_selected]

        # add the closest distance to selected point
        is_closer = D[I] < dist_to_closest_selected
        dist_to_closest_selected[is_closer] = D[I][is_closer]

        # add it to the selected and remove from not selected
        selected.append(I)
        not_selected = np.delete(not_selected, index_in_not_selected)

    return selected
