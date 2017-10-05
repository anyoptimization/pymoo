import numpy as np

from configuration import Configuration
from util.misc import print_pop
from util.non_dominated_rank import NonDominatedRank


class RankAndCrowdingSurvival:
    def survive(self, pop, size):
        # calculate rank and crowding and sort accordingly
        rank, crowding = RankAndCrowdingSurvival.calc_rank_and_crowding(pop)
        sorted_idx = sorted(range(len(pop)), key=lambda x: (rank[x], -crowding[x]))
        pop = [pop[i] for i in sorted_idx]

        #print_pop(pop, rank, crowding, sorted_idx, 20)

        # now truncate the population
        return pop[:size]



    @staticmethod
    def calc_rank_and_crowding(pop):
        fronts = NonDominatedRank.calc_as_fronts_pygmo(pop)
        # fronts = NonDominatedRank.calc_as_fronts(pop)
        rank = NonDominatedRank.calc_from_fronts(fronts)
        cd = np.zeros(len(pop))
        for front in fronts:
            cd_of_front = RankAndCrowdingSurvival.calc_crowding_distance([pop[x] for x in front])
            for i, idx_of_individual in enumerate(front):
                cd[idx_of_individual] = cd_of_front[i]

        return rank, cd

    @staticmethod
    def calc_crowding_distance(front):
        n = len(front)

        if n == 0:
            return []
        elif n == 1 or n == 2:
            return np.full(n, np.inf)
        else:

            cd = np.zeros(n)
            n_obj = len(front[0].f)

            # for each objective
            for j in range(n_obj):

                # sort by its objective
                sorted_idx = sorted(range(n), key=lambda x: front[x].f[j])

                # set the corner points to infinity
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                # get min and max for normalization
                f_min = front[sorted_idx[0]].f[j]
                f_max = front[sorted_idx[-1]].f[j]

                # calculate the normalization - avoid division by 0
                norm = f_max - f_min if f_min != f_max else Configuration.EPS

                # add up the crowding measure for all points in between
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (front[sorted_idx[i+1]].f[j] - front[sorted_idx[i-1]].f[j]) / norm

        return cd
