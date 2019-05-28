import sys

import numpy as np
from pymoo.util import plotting
from scipy import special

from pymoo.util.misc import unique_rows, cdist
from pymoo.util.plotting import plot_3d


class ReferenceDirectionFactory:

    def __init__(self, n_dim, scaling=None, lexsort=False) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.scaling = scaling
        self.lexsort = lexsort

    def do(self):

        if self.n_dim == 1:
            return np.array([[1.0]])
        else:
            ref_dirs = self._do()
            if self.scaling is not None:
                ref_dirs = ref_dirs * self.scaling + ((1 - self.scaling) / self.n_dim)

            # do ref_dirs is desired
            if self.lexsort:
                I = np.lexsort([ref_dirs[:, j] for j in range(ref_dirs.shape[1])][::-1])
                ref_dirs = ref_dirs[I]

            return ref_dirs

    def _do(self):
        return None


def sample_on_simplex(n_points, n_dim, method="kraemer", seed=None):
    if seed is not None:
        np.random.seed(seed)

    if method == "random":
        # sample random points
        rnd = np.random.random((n_points, n_dim))
        rnd = rnd / rnd.sum(axis=1)[:, None]

    elif method == "kraemer":
        M = sys.maxsize
        _rnd = np.random.randint(0, M, (n_points, n_dim - 1))
        _rnd = np.column_stack([np.zeros(n_points), _rnd, np.full(n_points, M)])
        _rnd = np.sort(_rnd, axis=1)

        rnd = np.full((n_points, n_dim), np.nan)
        for i in range(1, n_dim + 1):
            rnd[:, i - 1] = _rnd[:, i] - _rnd[:, i - 1]
        rnd /= M

    elif method == "das_dennis":
        rnd = UniformReferenceDirectionFactory(n_dim, n_points=n_points).do()

    return rnd


def select_points_with_maximum_distance(X, n_select, selected=[]):
    n_points, n_dim = X.shape

    # calculate the distance matrix
    D = cdist(X, X)

    # if no selection provided pick randomly in the beginning
    if len(selected) == 0:
        selected = [np.random.randint(len(X))]

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


class SamplingReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim, scaling=None, n_points=None, n_sample_points=5000, sampling="kraemer", seed=1,
                 kmeans=True) -> None:
        super().__init__(n_dim, scaling)
        self.n_sample_points = n_sample_points
        self.sampling = sampling
        self.seed = seed
        self.kmeans = kmeans

        if n_points is None:
            raise Exception("Please provide the number of points to be factored!")
        self.n_points = n_points

    def _do(self):

        rnd = sample_on_simplex(self.n_sample_points, self.n_dim, method=self.sampling, seed=self.seed)

        # add the corner coordinates
        X = np.row_stack([np.eye(self.n_dim), rnd])

        selected = list(range(self.n_dim))
        I = select_points_with_maximum_distance(X, self.n_points, selected=selected)

        centroids = X[I]

        # now execute some iterations of k-means to refine the points
        n_max_iter = 10000
        a_tol = 1e-12

        # if clustering should be performed after this algorithm
        if self.kmeans:

            for i in range(n_max_iter):

                # assign all points to one of the centroids
                points_to_centroid = cdist(X, centroids).argmin(axis=1)

                centroids_to_points = [[] for _ in range(len(centroids))]
                for j, k in enumerate(points_to_centroid):
                    centroids_to_points[k].append(j)

                last_centroids = np.copy(centroids)
                for j in range(len(centroids_to_points)):
                    centroids[j] = np.mean(X[centroids_to_points[j]], axis=0)

                if np.abs(centroids - last_centroids).sum(axis=1).mean() < a_tol:
                    break

            # because the centroids went inside, we need to stretch the points finally
            index_of_extreme = centroids.argmax(axis=0)
            for i in range(self.n_dim):
                ext = np.copy(centroids[index_of_extreme[i]])
                ext[i] = 0
                centroids -= ext

                centroids[centroids < 0] = 0

        # make sure the sum is one after stretching
        centroids = centroids / centroids.sum(axis=1)[:, None]

        return centroids


class UniformReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim, scaling=None, n_points=None, n_partitions=None, **kwargs) -> None:

        super().__init__(n_dim, scaling=scaling, **kwargs)

        if n_points is not None:
            self.n_partitions = UniformReferenceDirectionFactory.get_partition_closest_to_points(n_points, n_dim)
        else:
            if n_partitions is None:
                raise Exception("Either provide number of partitions or number of points.")
            else:
                self.n_partitions = n_partitions

    def _do(self):
        return self.uniform_reference_directions(self.n_partitions, self.n_dim)

    @staticmethod
    def get_partition_closest_to_points(n_points, n_dim):

        # in this case the do method will always return one values anyway
        if n_dim == 1:
            return 0

        n_partitions = 1
        _n_points = UniformReferenceDirectionFactory.get_n_points(n_partitions, n_dim)
        while _n_points <= n_points:
            n_partitions += 1
            _n_points = UniformReferenceDirectionFactory.get_n_points(n_partitions, n_dim)
        return n_partitions - 1

    @staticmethod
    def get_n_points(n_partitions, n_dim):
        return int(special.binom(n_dim + n_partitions - 1, n_partitions))

    def uniform_reference_directions(self, n_partitions, n_dim):
        if n_partitions == 0:
            return np.full((1, n_dim), 1 / n_dim)
        else:
            ref_dirs = []
            ref_dir = np.full(n_dim, np.nan)
            self.__uniform_reference_directions(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
            return np.concatenate(ref_dirs, axis=0)

    def __uniform_reference_directions(self, ref_dirs, ref_dir, n_partitions, beta, depth):
        if depth == len(ref_dir) - 1:
            ref_dir[depth] = beta / (1.0 * n_partitions)
            ref_dirs.append(ref_dir[None, :])
        else:
            for i in range(beta + 1):
                ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
                self.__uniform_reference_directions(ref_dirs, np.copy(ref_dir), n_partitions, beta - i,
                                                    depth + 1)


class MultiLayerReferenceDirectionFactory:

    def __init__(self, *args) -> None:
        self.layers = args

    def add_layer(self, *args):
        self.layers.extend(args)

    def do(self):
        ref_dirs = []
        for factory in self.layers:
            ref_dirs.append(factory.do())
        ref_dirs = np.concatenate(ref_dirs, axis=0)

        return unique_rows(ref_dirs)


if __name__ == '__main__':
    ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=0).do()

    plotting.plot(ref_dirs)

    ref_dirs = SamplingReferenceDirectionFactory(3, n_points=200, n_sample_points=10000).do()
    print(np.sum(ref_dirs, axis=1))

    plotting.plot(ref_dirs)

    exit(1)

    n_dim = 9
    ref_dirs = MultiLayerReferenceDirectionFactory([
        UniformReferenceDirectionFactory(n_dim, n_partitions=3, scaling=1.0),
        UniformReferenceDirectionFactory(n_dim, n_partitions=4, scaling=0.9),
        UniformReferenceDirectionFactory(n_dim, n_partitions=2, scaling=0.8),
        UniformReferenceDirectionFactory(n_dim, n_partitions=2, scaling=0.7),
        UniformReferenceDirectionFactory(n_dim, n_partitions=2, scaling=0.6),
        UniformReferenceDirectionFactory(n_dim, n_partitions=2, scaling=0.5),
        UniformReferenceDirectionFactory(n_dim, n_partitions=2, scaling=0.4),
        UniformReferenceDirectionFactory(n_dim, n_partitions=2, scaling=0.3),
        UniformReferenceDirectionFactory(n_dim, n_partitions=2, scaling=0.2),
        UniformReferenceDirectionFactory(n_dim, n_partitions=2, scaling=0.1),
    ]).do()

    # ref_dirs = UniformReferenceDirectionFactory(3, n_points=100).do()

    # np.savetxt('ref_dirs_9.csv', ref_dirs)

    print(ref_dirs.shape)

    exit(1)

    multi_layer = MultiLayerReferenceDirectionFactory()
    multi_layer.add_layer(UniformReferenceDirectionFactory(10, n_partitions=2, scaling=0.5))
    multi_layer.add_layer(UniformReferenceDirectionFactory(10, n_partitions=3, scaling=1.0))

    # multi_layer.add_layer(0.5, UniformReferenceDirectionFactory(3, n_partitions=10))
    ref_dirs = multi_layer.do()

    print(UniformReferenceDirectionFactory.get_partition_closest_to_points(100, 3))
    print(ref_dirs.shape)

    plot_3d(ref_dirs)
