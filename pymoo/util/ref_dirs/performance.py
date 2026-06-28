"""Performance metrics for reference direction distributions."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.qhull import Delaunay
from scipy.stats import gmean

from pymoo.util.misc import (
    distance_of_closest_points_to_others,
    vectorized_cdist,
    cdist,
)
from pymoo.util.ref_dirs.das_dennis import DasDennis
from pymoo.util.reference_direction import get_partition_closest_to_points


def distance_of_closest_point(ref_dirs: NDArray) -> float:
    """Compute the minimum distance between any two reference directions.

    Args:
        ref_dirs: Array of reference direction vectors.

    Returns:
        The minimum distance to the closest point.
    """
    _, dist = distance_of_closest_points_to_others(ref_dirs)
    return dist.min()


def average_distance_to_other_points(ref_dirs: NDArray) -> float:
    """Compute the average pairwise distance between reference directions.

    Args:
        ref_dirs: Array of reference direction vectors.

    Returns:
        The average distance between all pairs of points.
    """
    D: NDArray = vectorized_cdist(ref_dirs, ref_dirs)  # type: ignore[assignment]
    D = D[np.triu_indices(len(ref_dirs), 1)]
    return float(D.mean())


def closest_point_variance(z: NDArray) -> float:
    """Compute variance of distances to closest points.

    Args:
        z: Array of points.

    Returns:
        The variance of closest point distances.
    """
    for row in np.eye(z.shape[1]):
        if not np.any(np.all(row == z, axis=1)):
            z = np.vstack([z, row])

    D: NDArray = vectorized_cdist(z, z)  # type: ignore[assignment]
    np.fill_diagonal(D, 1)

    return float(D.min(axis=1).var())


def closest_point_variance_mod(z: NDArray) -> float:
    """Compute modified variance of distances to k-th closest points.

    Args:
        z: Array of points.

    Returns:
        The variance of k-th nearest neighbor distances.
    """
    n_points, n_dim = z.shape

    for row in np.eye(z.shape[1]):
        if not np.any(np.all(row == z, axis=1)):
            z = np.vstack([z, row])

    D: NDArray = vectorized_cdist(z, z)  # type: ignore[assignment]
    np.fill_diagonal(D, np.inf)

    k = int(np.ceil(np.sqrt(n_dim)))
    I = D.argsort(axis=1)[:, k - 1]  # noqa: E741

    return float(D[np.arange(n_points), I].var())


def geometric_mean_var(z: NDArray) -> float:
    """Compute variance of geometric mean distances to k nearest neighbors.

    Args:
        z: Array of points.

    Returns:
        The variance of geometric mean distances.
    """
    for row in np.eye(z.shape[1]):
        if not np.any(np.all(row == z, axis=1)):
            z = np.vstack([z, row])
    n_points, n_dim = z.shape

    D: NDArray = vectorized_cdist(z, z)  # type: ignore[assignment]
    np.fill_diagonal(D, np.inf)

    k = n_dim - 1
    I = D.argsort(axis=1)[:, :k]  # noqa: E741

    first = np.column_stack([np.arange(n_points) for _ in range(k)])

    val = gmean(D[first, I], axis=1)

    return float(val.var())


def mean_mean(z: NDArray) -> float:
    """Compute mean of mean distances to k nearest neighbors.

    Args:
        z: Array of points.

    Returns:
        The mean of average nearest neighbor distances.
    """
    for row in np.eye(z.shape[1]):
        if not np.any(np.all(row == z, axis=1)):
            z = np.vstack([z, row])
    n_points, n_dim = z.shape

    D: NDArray = vectorized_cdist(z, z)  # type: ignore[assignment]
    np.fill_diagonal(D, np.inf)

    k = n_dim - 1
    I = D.argsort(axis=1)[:, :k]  # noqa: E741

    first = np.column_stack([np.arange(n_points) for _ in range(k)])

    val = np.mean(D[first, I], axis=1)

    return float(val.mean())


def potential_energy(x: NDArray) -> float:
    """Compute potential energy metric for point distribution.

    Args:
        x: Array of points.

    Returns:
        The potential energy value.
    """
    _x = x / x.sum(axis=1)[:, None]
    D = ((_x[:, None] - _x[None, :]) ** 2).sum(axis=2)
    D = D[np.triu_indices(len(_x), 1)]
    return (1 / D).mean()


def iterative_igd(X: NDArray, n_partitions: int | None = None, batch_size: int = 100) -> float:
    """Compute inverted generational distance using iterative batch evaluation.

    Args:
        X: Array of points.
        n_partitions: Number of partitions for reference set (auto if None).
        batch_size: Number of reference points per batch.

    Returns:
        The inverted generational distance value.
    """
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


def gram_schmidt(X: NDArray, row_vecs: bool = True, norm: bool = True) -> NDArray:
    """Orthonormalize vectors using Gram-Schmidt process.

    Args:
        X: Array of vectors.
        row_vecs: If True, rows are vectors; if False, columns are vectors.
        norm: If True, normalize the result.

    Returns:
        Orthonormalized vectors.
    """
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


def triangulation(X: NDArray) -> float:
    """Compute triangulation metric for point distribution.

    Args:
        X: Array of points on a simplex.

    Returns:
        The triangulation metric value.
    """
    return simplex_edge_difference(project_onto_one_dim_less(X))


def project_onto_one_dim_less(X: NDArray) -> NDArray:
    """Project points onto a subspace one dimension less.

    Args:
        X: Array of points in n dimensions.

    Returns:
        Points projected onto an (n-1)-dimensional subspace.
    """
    E = np.eye(X.shape[1])
    P = E[0]
    D = E[1:] - P
    O = gram_schmidt(D)  # noqa: E741
    return (X - P) @ O.T


def simplex_edge_difference(X: NDArray) -> float:
    """Compute mean edge difference from Delaunay triangulation.

    Args:
        X: Array of points.

    Returns:
        The mean edge difference metric.
    """
    tri = Delaunay(X, qhull_options="QJ")

    simplices = []
    for e in tri.simplices:
        diff = X[e[1:]] - X[e[0]]
        det = np.linalg.det(diff)
        if det > 1e-6:
            simplices.append(e)

    val_list: list[float] = []
    for triangle in simplices:
        dists = np.zeros(len(triangle))

        for i in range(len(triangle)):
            a, b = triangle[i], triangle[(i + 1) % len(triangle)]
            dists[i] = np.linalg.norm(X[a] - X[b])

        val_list.append(float(dists.max() - dists.min()))

    val: NDArray = np.array(val_list)

    return float(val.mean())
