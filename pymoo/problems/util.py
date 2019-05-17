import os

import numpy as np
import autograd.numpy as anp



def load_pareto_front_from_file(fname):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(current_dir, "pf", "%s" % fname)
    if os.path.isfile(fname):
        return anp.loadtxt(fname)


def get_uniform_weights(n_points, n_dim):
    return UniformReferenceDirectionFactory(n_dim, n_points=n_points).do()


def binomial(n, k):
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class UniformReferenceDirectionFactory:

    def __init__(self, n_dim, n_points=None, n_partitions=None) -> None:

        self.n_dim = n_dim
        if n_points is not None:
            self.n_partitions = UniformReferenceDirectionFactory.get_partition_closest_to_points(n_points, n_dim)
        else:
            if n_partitions is None:
                raise Exception("Either provide number of partitions or number of points.")
            else:
                self.n_partitions = n_partitions

    def do(self):
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
        return int(binomial(n_dim + n_partitions - 1, n_partitions))

    def uniform_reference_directions(self, n_partitions, n_dim):
        ref_dirs = []
        ref_dir = anp.full(n_dim, anp.inf)
        self.__uniform_reference_directions(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return anp.concatenate(ref_dirs, axis=0)

    def __uniform_reference_directions(self, ref_dirs, ref_dir, n_partitions, beta, depth):
        if depth == len(ref_dir) - 1:
            ref_dir[depth] = beta / (1.0 * n_partitions)
            ref_dirs.append(ref_dir[None, :])
        else:
            for i in range(beta + 1):
                ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
                self.__uniform_reference_directions(ref_dirs, anp.copy(ref_dir), n_partitions, beta - i,
                                                    depth + 1)