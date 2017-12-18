import numpy as np

from pymoo.configuration import Configuration
from pymoo.model.survival import Survival
from pymoo.util.non_dominated_rank import NonDominatedRank


class RankAndCrowdingSurvival(Survival):
    def _do(self, pop, size):
        # calculate rank and crowding and sort accordingly
        rank, crowding = RankAndCrowdingSurvival.calc_rank_and_crowding(pop)
        sorted_idx = sorted(range(pop.size()), key=lambda x: (rank[x], -crowding[x]))

        # now truncate the population
        sorted_idx = sorted_idx[:size]
        pop.filter(sorted_idx)

        return pop

    @staticmethod
    def calc_rank_and_crowding(pop):
        fronts = NonDominatedRank.calc_as_fronts(pop.F, pop.G)
        rank = NonDominatedRank.calc_from_fronts(fronts)
        cd = np.zeros(pop.size())
        for front in fronts:
            cd[front] = RankAndCrowdingSurvival.crowding_distance(pop.F[front, :])

        return rank, cd

    @staticmethod
    def crowding_distance(front):

        n = front.shape[0]

        if n == 0:
            return []
        elif n == 1 or n == 2:
            return np.full(n, np.inf)
        else:

            cd = np.zeros(n)
            n_obj = front.shape[1]

            f_min = np.min(front, axis=0)
            f_max = np.max(front, axis=0)

            # for each objective
            for j in range(n_obj):

                # sort by its objective
                sorted_idx = sorted(list(range(n)), key=lambda x: front[x, j])

                # set the corner points to infinity
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                # calculate the normalization - avoid division by 0
                norm = f_max[j] - f_min[j] if f_min[j] != f_max[j] else Configuration.EPS

                # add up the crowding measure for all points in between
                for i in range(1, n - 1):
                    if np.isinf(cd[sorted_idx[i]]):
                        continue
                    else:
                        last = front[sorted_idx[i - 1], j]
                        current = front[sorted_idx[i], j]
                        next = front[sorted_idx[i + 1], j]
                        # this eliminates duplicates since they have a crowding distance of 0.
                        if last != current:
                            cd[sorted_idx[i]] += (next - last) / norm

        cd = [val / n_obj for val in cd]
        return cd
