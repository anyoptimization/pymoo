"""Reduction-based reference direction factory using k-means clustering."""

import numpy as np
from numpy.typing import NDArray

from pymoo.util.misc import cdist
from pymoo.util.ref_dirs.misc import project_onto_unit_simplex_recursive
from pymoo.util.reference_direction import (
    ReferenceDirectionFactory,
    sample_on_unit_simplex,
    select_points_with_maximum_distance,
    get_partition_closest_to_points,
    UniformReferenceDirectionFactory,
)


def kmeans(
    X: NDArray,
    centroids: NDArray,
    n_max_iter: int,
    a_tol: float,
    n_ignore: int,
) -> NDArray:
    """Perform k-means clustering on the unit simplex.

    Args:
        X: Data points to cluster.
        centroids: Initial centroids, modified in-place.
        n_max_iter: Maximum number of iterations.
        a_tol: Convergence tolerance.
        n_ignore: Number of initial centroids to keep fixed.

    Returns:
        The final centroids.
    """
    for i in range(n_max_iter):
        # copy the old centroids
        last_centroids = np.copy(centroids)

        # assign all points to one of the centroids
        points_to_centroid = cdist(X, centroids).argmin(axis=1)

        centroids_to_points: list[list[int]] = [[] for _ in range(len(centroids))]
        for j, k in enumerate(points_to_centroid):
            centroids_to_points[k].append(j)

        for j in range(n_ignore, len(centroids_to_points)):
            centroids[j] = np.mean(X[centroids_to_points[j]], axis=0)

        project_onto_unit_simplex_recursive(centroids)
        centroids /= centroids.sum(axis=1)[:, None]

        delta = np.abs(centroids - last_centroids).sum(axis=1).mean()

        if delta < a_tol:
            break

    return centroids


class ReductionBasedReferenceDirectionFactory(ReferenceDirectionFactory):
    """Factory for generating reference directions using reduction and k-means."""

    def __init__(
        self,
        n_dim: int,
        n_points: int | None,
        scaling: float | None = None,
        n_sample_points: int = 10000,
        sampling: str = "kraemer",
        kmeans: bool = True,
        kmeans_max_iter: int = 1000,
        kmeans_a_tol: float = 0.0001,
        **kwargs,  # type: ignore[no-untyped-def]
    ) -> None:
        """Initialize the reduction-based reference direction factory.

        Args:
            n_dim: Number of objectives.
            n_points: Number of reference directions to generate.
            scaling: Scaling factor for the reference directions.
            n_sample_points: Number of points to sample from the simplex.
            sampling: Sampling method ("kraemer" or other).
            kmeans: Whether to apply k-means refinement.
            kmeans_max_iter: Maximum iterations for k-means.
            kmeans_a_tol: Convergence tolerance for k-means.
            **kwargs: Additional arguments for parent class.

        Raises:
            Exception: If n_points is None.
        """
        super().__init__(n_dim, scaling, **kwargs)
        self.n_sample_points = n_sample_points
        self.sampling = sampling
        self.kmeans = kmeans
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_a_tol = kmeans_a_tol

        if n_points is None:
            raise Exception("Please provide the number of points to be factored!")

        self.n_points = n_points

    def _do(self, random_state: int | None = None) -> NDArray:
        """Generate reference directions.

        Args:
            random_state: Random seed for reproducibility.

        Returns:
            Array of reference directions.
        """
        rnd = sample_on_unit_simplex(
            self.n_sample_points,
            self.n_dim,
            random_state=random_state,
            unit_simplex_mapping=self.sampling,
        )

        def h(n: int) -> int:
            return get_partition_closest_to_points(n, self.n_dim)

        H = h(self.n_points)

        E = UniformReferenceDirectionFactory(self.n_dim, n_partitions=H).do()
        E = E[np.any(E == 0, axis=1)]

        # add the edge coordinates
        X = np.vstack([E, rnd])

        I = select_points_with_maximum_distance(  # noqa: E741
            X, self.n_points, selected=list(range(len(E)))
        )
        centroids = X[I].copy()

        if self.kmeans:
            centroids = kmeans(X, centroids, self.kmeans_max_iter, self.kmeans_a_tol, len(E))

        return centroids
