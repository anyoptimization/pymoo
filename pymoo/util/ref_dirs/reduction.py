import numpy as np


from pymoo.util.misc import cdist

from pymoo.util.ref_dirs.misc import project_onto_unit_simplex_recursive
from pymoo.util.reference_direction import ReferenceDirectionFactory, sample_on_unit_simplex, \
    select_points_with_maximum_distance, get_partition_closest_to_points, UniformReferenceDirectionFactory


def kmeans(X, centroids, n_max_iter, a_tol, n_ignore):

    for i in range(n_max_iter):

        # copy the old centroids
        last_centroids = np.copy(centroids)

        # assign all points to one of the centroids
        points_to_centroid = cdist(X, centroids).argmin(axis=1)

        centroids_to_points = [[] for _ in range(len(centroids))]
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

    def __init__(self,
                 n_dim,
                 n_points,
                 scaling=None,
                 n_sample_points=10000,
                 sampling="kraemer",
                 kmeans=True,
                 kmeans_max_iter=1000,
                 kmeans_a_tol=0.0001,
                 **kwargs):

        super().__init__(n_dim, scaling, **kwargs)
        self.n_sample_points = n_sample_points
        self.sampling = sampling
        self.kmeans = kmeans
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_a_tol = kmeans_a_tol

        if n_points is None:
            raise Exception("Please provide the number of points to be factored!")

        self.n_points = n_points

    def _do(self, random_state=None):
        rnd = sample_on_unit_simplex(self.n_sample_points, self.n_dim, random_state=random_state, unit_simplex_mapping=self.sampling)

        def h(n):
            return get_partition_closest_to_points(n, self.n_dim)

        H = h(self.n_points)

        E = UniformReferenceDirectionFactory(self.n_dim, n_partitions=H).do()
        E = E[np.any(E == 0, axis=1)]

        # add the edge coordinates
        X = np.vstack([E, rnd])

        I = select_points_with_maximum_distance(X, self.n_points, selected=list(range((len(E)))))
        centroids = X[I].copy()

        if self.kmeans:
            #centroids = kmeans(X, centroids, self.kmeans_max_iter, self.kmeans_a_tol, 0)
            centroids = kmeans(X, centroids, self.kmeans_max_iter, self.kmeans_a_tol, len(E))

        return centroids
