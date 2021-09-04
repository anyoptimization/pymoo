import numpy as np
from scipy.spatial.ckdtree import cKDTree

from pymoo.core.indicator import Indicator


class DecisionMaking(Indicator):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.default_if_empty = None

    def _do(self, F, *args, **kwargs):
        pass


class NeighborFinder:

    def __init__(self, N,
                 epsilon=0.125,
                 n_neighbors=None,
                 n_min_neigbors=None,
                 consider_2d=True):

        super().__init__()
        self.N = N
        self.consider_2d = consider_2d

        _, n_dim = N.shape

        # at least find min(dimensionality times two neighbors, number PO solutions - 1) - if enabled
        if n_min_neigbors == "auto":
            self.n_min_neigbors = min(2 * n_dim, _ - 1)

        # disable the minimum neighbor variable
        else:
            self.n_min_neigbors = np.inf

        # either choose epsilon
        self.epsilon = epsilon

        # if none choose the number of neighbors
        self.n_neighbors = n_neighbors

        if self.N.shape[1] == 1:
            raise Exception("At least 2 objectives must be provided.")

        elif self.consider_2d and self.N.shape[1] == 2:
            self.min, self.max = N.min(), N.max()
            self.rank = np.argsort(N[:, 0])
            self.pos_in_rank = np.argsort(self.rank)

        else:
            self.tree = cKDTree(N)

    def find(self, i):

        if self.consider_2d and self.N.shape[1] == 2:
            neighbours = []

            pos = self.pos_in_rank[i]
            if pos > 0:
                neighbours.append(self.rank[pos - 1])
            if pos < len(self.N) - 1:
                neighbours.append(self.rank[pos + 1])

        else:

            # for each neighbour in a specific radius of that solution
            if self.epsilon is not None:
                neighbours = self.tree.query_ball_point([self.N[i]], self.epsilon).tolist()[0]
            elif self.n_neighbors is not None:
                neighbours = self.tree.query([self.N[i]], k=self.n_neighbors + 1)[1].tolist()[0]
            else:
                raise Exception("Either define epsilon or number of neighbors.")

            # in case n_min_neigbors is enabled
            if len(neighbours) < self.n_min_neigbors:
                neighbours = self.tree.query([self.N[i]], k=self.n_min_neigbors + 1)[1].tolist()[0]

        return neighbours


def find_outliers_upper_tail(mu):

    # remove values that are nan
    I = np.where(np.logical_and(np.logical_not(np.isnan(mu)), np.logical_not(np.isinf(mu))))[0]
    mu = mu[I]

    # calculate mean and sigma
    mean, sigma = mu.mean(), mu.std()

    # calculate the deviation in terms of sigma
    deviation = (mu - mean) / sigma

    # 2 * sigma is considered as an outlier
    S = I[np.where(deviation >= 2)[0]]

    if len(S) == 0 and deviation.max() > 1:
        S = I[[np.argmax(mu)]]

    return S if len(S) > 0 else None
