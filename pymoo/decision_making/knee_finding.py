import numpy as np
from scipy.spatial import cKDTree

from pymoo.util.normalization import normalize as normalize_by_bounds
from pymoo.decision_making.decision_making import DecisionMaking, normalize


class KneeFinding(DecisionMaking):

    def __init__(self, epsilon=0.125, penalize_extremes=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.penalize_extremes = penalize_extremes

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
            if len(neighbours) < m + 1:
                neighbours = tree.query([F[i]], k=m + 1)[1].tolist()[0]

            # now let us calculate the trade-off from each solution to another
            tradeoff = np.full(len(neighbours), np.inf)

            for j, neighbour in enumerate(neighbours):

                # calculate the gain and loss
                gain, loss = 0.0, 0.0
                for k in range(m):
                    gain = gain + max(0, F[neighbour][k] - F[i][k])
                    loss = loss + max(0, F[i][k] - F[neighbour][k])

                # only sum up if there is a loss
                if loss > 0:
                    tradeoff[j] = gain / loss

            # if at least one trade-off was found
            if np.any(np.isinf(tradeoff)):
                mu[i] = tradeoff.min()

                # penalize the extreme points by dividing through the range of the point itself
                if self.penalize_extremes:
                    norm = F[i].max() - F[i].min()
                    if norm > 0:
                        mu[i] /= norm
        I = np.where(np.logical_and(mu > 0, np.logical_not(np.isnan(mu))))[0]
        mu = normalize_by_bounds(mu[I])

        split = mu.mean() + 1.5 * mu.std()
        selected = np.where(mu > split)[0]

        return I[selected]
