import numpy as np

from pymoo.model.decision_making import DecisionMaking, normalize, NeighborFinder, find_outliers_upper_tail


class HighTradeoffPointsInverted(DecisionMaking):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _do(self, F, **kwargs):

        n, m = F.shape

        if self.normalize:
            F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, n_min_neigbors="auto")

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # neighbors to the current point
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmean(tradeoff)

        return find_outliers_upper_tail(mu)

if __name__ == '__main__':

    F = np.array([[2, 13], [3, 11], [7, 9], [9, 4], [12, 3]])
    #_, w = PseudoWeights(np.array([0.5, 0.5])).do(F, return_pseudo_weights=True)
    HighTradeoffPointsInverted(normalize=False).do(F)
