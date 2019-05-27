import numpy as np
from scipy import special

from pymoo.util.misc import unique_rows
from pymoo.util.plotting import plot_3d


class ReferenceDirectionFactory:

    def __init__(self, n_dim, scaling=None) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.scaling = scaling

    def do(self):

        if self.n_dim == 1:
            return np.array([[1.0]])
        else:
            ref_dirs = self._do()
            if self.scaling is not None:
                ref_dirs = ref_dirs * self.scaling + ((1 - self.scaling) / self.n_dim)
            return ref_dirs

    def _do(self):
        return None


class UniformReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim, scaling=None, n_points=None, n_partitions=None) -> None:

        super().__init__(n_dim, scaling=scaling)

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
        ref_dirs = []
        ref_dir = np.full(n_dim, np.inf)
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

    def __init__(self, layers=[]) -> None:
        self.layers = layers

    def add_layer(self, factory):
        self.layers.append(factory)

    def do(self):
        ref_dirs = []
        for factory in self.layers:
            ref_dirs.append(factory.do())
        ref_dirs = np.concatenate(ref_dirs, axis=0)

        return unique_rows(ref_dirs)


if __name__ == '__main__':
    ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()
    print(np.sum(ref_dirs, axis=1))

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

    #np.savetxt('ref_dirs_9.csv', ref_dirs)

    print(ref_dirs.shape)

    exit(0)

    multi_layer = MultiLayerReferenceDirectionFactory()
    multi_layer.add_layer(UniformReferenceDirectionFactory(10, n_partitions=2, scaling=0.5))
    multi_layer.add_layer(UniformReferenceDirectionFactory(10, n_partitions=3, scaling=1.0))

    # multi_layer.add_layer(0.5, UniformReferenceDirectionFactory(3, n_partitions=10))
    ref_dirs = multi_layer.do()

    print(UniformReferenceDirectionFactory.get_partition_closest_to_points(100, 3))
    print(ref_dirs.shape)

    plot_3d(ref_dirs)
