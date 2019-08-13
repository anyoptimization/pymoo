import numpy as np
from scipy.spatial import cKDTree

from pymoo.model.decision_making import DecisionMaking, normalize


class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _do(self, F, **kwargs):

        n, m = F.shape

        if self.normalize:
            F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        tree = cKDTree(F)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbours = tree.query_ball_point([F[i]], self.epsilon).tolist()[0]

            # consider at least m+1 neighbours - if not found force it
            if len(neighbours) < 2 * m + 1:
                neighbours = tree.query([F[i]], k=m + 1)[1].tolist()[0]

            # calculate the trade-off to all neighbours
            diff = F[neighbours] - F[i]

            np.warnings.filterwarnings('ignore')
            tradeoff = np.maximum(0, diff).sum(axis=1) / np.maximum(0, -diff).sum(axis=1)

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)

        # remove values that are nan
        I = np.where(np.logical_not(np.isnan(mu)))[0]
        mu = mu[I]

        # calculate mean and sigma
        mean, sigma = mu.mean(), mu.std()

        # calculate the deviation in terms of sigma
        deviation = (mu - mean) / sigma

        # 2 * sigma is considered as an outlier
        S = I[np.where(deviation >= 2)[0]]

        if len(S) == 0 and deviation.max() > 1:
            S = I[np.argmax(mu)]

        if len(S) == 0:
            return None
        else:
            return S
