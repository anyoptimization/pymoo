from random import shuffle

import numpy as np

from configuration import Configuration
from rand.default_random_generator import DefaultRandomGenerator
from util.non_dominated_rank import NonDominatedRank
from util.quicksort import quicksort


class RankAndCrowdingSurvival:
    def survive(self, pop, size, rnd=DefaultRandomGenerator()):
        #fronts = NonDominatedRank.calc_as_fronts(pop)
        fronts = NonDominatedRank.calc_as_fronts_naive(pop)

        # contains all indices that survive
        next_pop = []
        rank = np.zeros(len(pop))
        crowding = np.zeros(len(pop))

        for front, n_front in zip(fronts, range(len(fronts))):

            # this is only necessary to have the same result as the C code
            #shuffle(front)

            #for i in front:
            #    print(pop[i].f[0])

            # set the rank of the individual
            for idx in front:
                rank[idx] = n_front

            # calculate the crowding and assign
            cd_of_front = RankAndCrowdingSurvival.calc_crowding_distance([pop[x] for x in front], rnd=rnd)
            for i, idx_of_individual in enumerate(front):
                crowding[idx_of_individual] = cd_of_front[i]

            # add the last front sorted according crowding dist
            if len(next_pop) + len(front) > size:
                sorted_idx = quicksort(list(front), key=lambda x: -crowding[x], rnd=rnd)
                next_pop.extend(sorted_idx[:size - len(next_pop)])
                break
            else:
                next_pop.extend(front)

        pop = [pop[i] for i in next_pop]
        rank = [rank[i] for i in next_pop]
        crowding = [crowding[i] for i in next_pop]

        #rank = [r + 1 for r in rank]
        #for i in range(len(pop)):
        #    print(i, pop[i].f, rank[i], crowding[i])
        #print('---------')

        # print_pop(pop, rank, crowding, sorted_idx, 20)

        # now truncate the population
        return pop, rank, crowding

    @staticmethod
    def calc_rank_and_crowding(pop, rnd=DefaultRandomGenerator()):
        #fronts = NonDominatedRank.calc_as_fronts_pygmo(pop)
        fronts = NonDominatedRank.calc_as_fronts_naive(pop)
        rank = NonDominatedRank.calc_from_fronts(fronts)
        cd = np.zeros(len(pop))
        for front in fronts:
            front.reverse()
            cd_of_front = RankAndCrowdingSurvival.calc_crowding_distance([pop[x] for x in front], rnd=rnd)
            for i, idx_of_individual in enumerate(front):
                cd[idx_of_individual] = cd_of_front[i]

        if len(pop) > 0:
            m = len(pop[0].f)
            cd = [val / m for val in cd]

        return rank, cd

    @staticmethod
    def calc_crowding_distance(front, rnd=DefaultRandomGenerator()):
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
                sorted_idx = quicksort(list(range(n)), key=lambda x: front[x].f[j], rnd=rnd)

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
                    if np.isinf(cd[sorted_idx[i]]):
                        continue
                    else:
                        last = front[sorted_idx[i - 1]].f[j]
                        current = front[sorted_idx[i]].f[j]
                        next = front[sorted_idx[i + 1]].f[j]
                        # this eliminates duplicates since they have a crowding distance of 0.
                        if last != current:
                            cd[sorted_idx[i]] += (next - last) / norm

        cd = [val / n_obj for val in cd]
        return cd
