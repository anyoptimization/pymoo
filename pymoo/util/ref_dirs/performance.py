import autograd.numpy as anp
import numpy as np
from scipy.spatial.qhull import Delaunay
from scipy.stats import gmean

from pymoo.util.misc import distance_of_closest_points_to_others, vectorized_cdist, cdist
from pymoo.util.ref_dirs.das_dennis import DasDennis
from pymoo.util.reference_direction import get_partition_closest_to_points


def distance_of_closest_point(ref_dirs):
    _, dist = distance_of_closest_points_to_others(ref_dirs)
    return dist.min()


def average_distance_to_other_points(ref_dirs):
    D = vectorized_cdist(ref_dirs, ref_dirs)
    D = D[np.triu_indices(len(ref_dirs), 1)]
    return D.mean()


def closest_point_variance(z):
    for row in np.eye(z.shape[1]):
        if not np.any(np.all(row == z, axis=1)):
            z = np.row_stack([z, row])

    D = vectorized_cdist(z, z)
    np.fill_diagonal(D, 1)

    return D.min(axis=1).var()


def closest_point_variance_mod(z):
    n_points, n_dim = z.shape

    for row in np.eye(z.shape[1]):
        if not np.any(np.all(row == z, axis=1)):
            z = np.row_stack([z, row])

    D = vectorized_cdist(z, z)
    np.fill_diagonal(D, np.inf)

    k = int(np.ceil(np.sqrt(n_dim)))
    I = D.argsort(axis=1)[:, k - 1]

    return D[np.arange(n_points), I].var()


def geometric_mean_var(z):
    for row in np.eye(z.shape[1]):
        if not np.any(np.all(row == z, axis=1)):
            z = np.row_stack([z, row])
    n_points, n_dim = z.shape

    D = vectorized_cdist(z, z)
    np.fill_diagonal(D, np.inf)

    k = n_dim - 1
    I = D.argsort(axis=1)[:, :k]

    first = np.column_stack([np.arange(n_points) for _ in range(k)])

    val = gmean(D[first, I], axis=1)

    return val.var()


def mean_mean(z):
    for row in np.eye(z.shape[1]):
        if not np.any(np.all(row == z, axis=1)):
            z = np.row_stack([z, row])
    n_points, n_dim = z.shape

    D = vectorized_cdist(z, z)
    np.fill_diagonal(D, np.inf)

    k = n_dim - 1
    I = D.argsort(axis=1)[:, :k]

    first = np.column_stack([np.arange(n_points) for _ in range(k)])

    val = np.mean(D[first, I], axis=1)

    return val.mean()


def potential_energy(x):
    _x = x / x.sum(axis=1)[:, None]
    D = ((_x[:, None] - _x[None, :]) ** 2).sum(axis=2)
    D = D[anp.triu_indices(len(_x), 1)]
    return (1 / D).mean()


def iterative_igd(X, n_partitions=None, batch_size=100):
    n_points, n_dim = X.shape

    if n_partitions is None:
        n_partitions = get_partition_closest_to_points(n_points * n_dim * 10, n_dim) + 1

    scaling = 1 + 1 / 2

    dd = DasDennis(n_partitions, n_dim, scaling=scaling)
    val = 0

    while dd.has_next():
        points = dd.next(n_points=batch_size)
        val += cdist(points, X).min(axis=1).sum()

    val /= dd.number_of_points()
    return val


def gram_schmidt(X, row_vecs=True, norm=True):
    if not row_vecs:
        X = X.T
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


def triangulation(X):
    return simplex_edge_difference(project_onto_one_dim_less(X))


def project_onto_one_dim_less(X):
    E = np.eye(X.shape[1])
    P = E[0]
    D = E[1:] - P
    O = gram_schmidt(D)
    return (X - P) @ O.T


def simplex_edge_difference(X):
    tri = Delaunay(X, qhull_options="QJ")

    simplices = []
    for e in tri.simplices:
        diff = X[e[1:]] - X[e[0]]
        det = np.linalg.det(diff)
        if det > 1e-6:
            simplices.append(e)

    val = []
    for triangle in simplices:
        dists = np.zeros(len(triangle))

        for i in range(len(triangle)):
            a, b = triangle[i], triangle[(i + 1) % len(triangle)]
            dists[i] = np.linalg.norm(X[a] - X[b])

        val.append(dists.max() - dists.min())

    val = np.array(val)

    return val.mean()
