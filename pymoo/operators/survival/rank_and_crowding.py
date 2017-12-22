import numpy as np

from pymoo.model.survival import Survival
from pymoo.util.misc import normalize
from pymoo.util.non_dominated_rank import NonDominatedRank


class RankAndCrowdingSurvival(Survival):
    def _do(self, pop, size, data):

        fronts = NonDominatedRank.calc_as_fronts(pop.F, pop.G)
        data.rank = NonDominatedRank.calc_from_fronts(fronts)
        data.crowding = np.zeros(pop.F.shape[0])

        for front in fronts:
            cd_of_front = RankAndCrowdingSurvival.calc_crowding_distance(pop.F[front, :])
            data.crowding[front] = cd_of_front

        sorted_idx = sorted(range(pop.size()), key=lambda x: (data.rank[x], -data.crowding[x]))

        # now truncate the population
        sorted_idx = sorted_idx[:size]
        pop.filter(sorted_idx)
        data.rank = data.rank[sorted_idx]
        data.crowding = data.crowding[sorted_idx]

        return pop

    @staticmethod
    def calc_crowding_distance(F, F_min=None, F_max=None):

        n = F.shape[0]
        m = F.shape[1]

        if n == 0:
            return []
        elif n == 1 or n == 2:
            return np.full(n, np.inf)
        else:

            cd = np.zeros(n)
            N = normalize(F, F_min, F_max)

            # for each objective
            for j in range(m):

                # sort by its objective
                sorted_idx = np.argsort(N[:, j])

                # set the corner points to infinity
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                # add up the crowding measure for all points in between
                for i in range(1, n - 1):
                    if np.isinf(cd[sorted_idx[i]]):
                        continue
                    else:
                        cd[sorted_idx[i]] += N[sorted_idx[i + 1], j] - N[sorted_idx[i - 1], j]

        return cd
