import warnings

import numpy as np

from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder


class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _do(self, F, **kwargs):
        n, m = F.shape

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)

        return find_outliers_upper_tail(mu)
