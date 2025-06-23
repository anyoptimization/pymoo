import numpy as np
from scipy.spatial.distance import pdist, squareform
from pymoo.util.misc import find_duplicates
from pymoo.functions import load_function


def get_crowding_function(label):

    if label == "cd":
        fun = FunctionalDiversity(calc_crowding_distance, filter_out_duplicates=False)
    elif (label == "pcd") or (label == "pruning-cd"):
        fun = FunctionalDiversity(load_function("calc_pcd"), filter_out_duplicates=True)
    elif label == "ce":
        fun = FunctionalDiversity(calc_crowding_entropy, filter_out_duplicates=True)
    elif label == "mnn":
        fun = FuncionalDiversityMNN(load_function("calc_mnn"), filter_out_duplicates=True)
    elif label == "2nn":
        fun = FuncionalDiversityMNN(load_function("calc_2nn"), filter_out_duplicates=True)
    elif hasattr(label, "__call__"):
        fun = FunctionalDiversity(label, filter_out_duplicates=True)
    elif isinstance(label, CrowdingDiversity):
        fun = label
    else:
        raise KeyError("Crowding function not defined")
    return fun


class CrowdingDiversity:

    def do(self, F, n_remove=0):
        # Converting types Python int to Cython int would fail in some cases converting to long instead
        n_remove = np.intc(n_remove)
        F = np.array(F, dtype=np.double)
        return self._do(F, n_remove=n_remove)

    def _do(self, F, n_remove=None):
        pass


class FunctionalDiversity(CrowdingDiversity):

    def __init__(self, function=None, filter_out_duplicates=True):
        self.function = function
        self.filter_out_duplicates = filter_out_duplicates
        super().__init__()

    def _do(self, F, **kwargs):

        n_points, n_obj = F.shape

        if n_points <= 2:
            return np.full(n_points, np.inf)

        else:

            if self.filter_out_duplicates:
                # filter out solutions which are duplicates - duplicates get a zero finally
                is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
            else:
                # set every point to be unique without checking it
                is_unique = np.arange(n_points)

            # index the unique points of the array
            _F = F[is_unique]

            _d = self.function(_F, **kwargs)

            d = np.zeros(n_points)
            d[is_unique] = _d

        return d


class FuncionalDiversityMNN(FunctionalDiversity):

    def _do(self, F, **kwargs):

        n_points, n_obj = F.shape

        if n_points <= n_obj:
            return np.full(n_points, np.inf)

        else:
            return super()._do(F, **kwargs)


def calc_crowding_distance(F, **kwargs):
    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.vstack([F, np.full(n_obj, np.inf)]) - np.vstack([np.full(n_obj, -np.inf), F])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

    # if we divide by zero because all values in one columns are equal replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(I, axis=0)
    cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    return cd


def calc_crowding_entropy(F, **kwargs):
    """Wang, Y.-N., Wu, L.-H. & Yuan, X.-F., 2010. Multi-objective self-adaptive differential 
    evolution with elitist archive and crowding entropy-based diversity measure. 
    Soft Comput., 14(3), pp. 193-209.

    Parameters
    ----------
    F : 2d array like
        Objective functions.

    Returns
    -------
    ce : 1d array
        Crowding Entropies
    """
    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.vstack([F, np.full(n_obj, np.inf)]) - np.vstack([np.full(n_obj, -np.inf), F])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dl = dist.copy()[:-1]
    du = dist.copy()[1:]

    # Fix nan
    dl[np.isnan(dl)] = 0.0
    du[np.isnan(du)] = 0.0

    # Total distance
    cd = dl + du

    # Get relative positions
    pl = (dl[1:-1] / cd[1:-1])
    pu = (du[1:-1] / cd[1:-1])

    # Entropy
    entropy = np.vstack([np.full(n_obj, np.inf),
                         -(pl * np.log2(pl) + pu * np.log2(pu)),
                         np.full(n_obj, np.inf)])

    # Crowding entropy
    J = np.argsort(I, axis=0)
    _cej = cd[J, np.arange(n_obj)] * entropy[J, np.arange(n_obj)] / norm
    _cej[np.isnan(_cej)] = 0.0
    ce = _cej.sum(axis=1)

    return ce


def calc_mnn_fast(F, **kwargs):
    return _calc_mnn_fast(F, F.shape[1], **kwargs)


def calc_2nn_fast(F, **kwargs):
    return _calc_mnn_fast(F, 2, **kwargs)


def _calc_mnn_fast(F, n_neighbors, **kwargs):

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = 1.0

    # F normalized
    F = (F - F.min(axis=0)) / norm

    # Distances pairwise (Inefficient)
    D = squareform(pdist(F, metric="sqeuclidean"))

    # M neighbors
    M = F.shape[1]
    _D = np.partition(D, range(1, M+1), axis=1)[:, 1:M+1]

    # Metric d
    d = np.prod(_D, axis=1)

    # Set top performers as np.inf
    _extremes = np.concatenate((np.argmin(F, axis=0), np.argmax(F, axis=0)))
    d[_extremes] = np.inf

    return d
