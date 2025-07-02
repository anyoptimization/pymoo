import numpy as np

from pymoo.util.reference_direction import ReferenceDirectionFactory


def check_n_points(n_points, n_dim):
    """
    Returns n_partitions or a numeric value associated with the exception message.
    """

    if n_dim == 1:
        return [0]
    
    I = n_dim * np.eye(n_dim)
    W = np.zeros((1, n_dim))
    edgeW = W
    i = 0

    while len(W) < n_points:
        edgeW = np.tile(edgeW, (n_dim, 1)) + np.repeat(I, edgeW.shape[0], axis=0)
        edgeW = np.unique(edgeW, axis=0)
        edgeW = edgeW [np.any(edgeW == 0, axis=1)]
        W = np.vstack((W + 1, edgeW))
        i += 1

    if len(W) == n_points:
        return [i]
    
    return  [len(W) - len(edgeW), i - 1, len(W), i]


def incremental_lattice(n_partitions, n_dim):
    I = n_dim * np.eye(n_dim)
    W = np.zeros((1, n_dim))
    edgeW = W

    for _ in range(n_partitions):
        edgeW = np.tile(edgeW, (n_dim, 1)) + np.repeat(I, edgeW.shape[0], axis=0)
        edgeW = np.unique(edgeW, axis=0)
        edgeW = edgeW [np.any(edgeW == 0, axis=1)]
        W = np.vstack((W + 1, edgeW))

    return W / (n_dim * n_partitions)

class IncrementalReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim, scaling=None, n_points=None, n_partitions=None, **kwargs) -> None:
        super().__init__(n_dim, scaling=scaling, **kwargs)

        if n_points is not None:
            results = check_n_points(n_points, n_dim)

            # the number of points are not matching to any partition number
            if len(results) > 1:
                raise Exception("The number of points (n_points = %s) can not be created uniformly.\n"
                                "Either choose n_points = %s (n_partitions = %s) or "
                                "n_points = %s (n_partitions = %s)." %
                                (n_points, results[0], results[1], results[2], results[3]))

            self.n_partitions = results[0]

        elif n_partitions is not None:
            self.n_partitions = n_partitions

        else:
            raise Exception("Either provide number of partitions or number of points.")

    def _do(self, **kwargs):
        return incremental_lattice(self.n_partitions, self.n_dim)
