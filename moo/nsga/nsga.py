import numpy as np

from moo.util.crowding_distance import calc_crowding_distance
from moo.util.individual import Individual
from moo.util.non_dominated_rank import NonDominatedRank

from pylab import *

class NSGA:
    def __init__(self):
        self.pop_size = 100
        self.factory = None
        self.crossover = None
        self.mutation = None
        self.n_gen = 100
        self.callback = None

    def solve(self, problem, n_gen, seed):

        # create the population according to the factoring strategy
        pop = self.factory.sample_more(self.pop_size)
        [ind.evaluate(problem) for ind in pop]

        # calculate rank and crowding and sort accordingly
        rank, crowding = NSGA.calc_rank_and_crowding(pop)
        sorted_idx = sorted(range(len(pop)), key=lambda x: (rank[x], -crowding[x]))
        pop = [pop[i] for i in sorted_idx]

        # for each generation
        for gen in range(n_gen):
            print 'gen = %d' % gen

            # create the mating pool
            perm = np.concatenate([np.random.permutation(self.pop_size), np.random.permutation(self.pop_size)])

            # do recombination and save offsprings
            offsprings = []
            for i in range(0, len(perm), 4):
                # find the parents by doing tournament selection
                parent1 = pop[min(perm[i], perm[i + 1])]
                parent2 = pop[min(perm[i + 2], perm[i + 3])]

                # do the crossover and mutation
                child1, child2 = self.crossover.crossover(parent1.x, parent2.x)
                child1 = self.mutation.mutate(child1)
                child2 = self.mutation.mutate(child2)

                offsprings.append(Individual(child1))
                offsprings.append(Individual(child2))

            # evaluate the offspring population
            [ind.evaluate(problem) for ind in offsprings]

            # merge the population and offsprings
            pop = pop + offsprings

            # calculate rank and crowding and sort accordingly
            rank, crowding = NSGA.calc_rank_and_crowding(pop)
            sorted_idx = sorted(range(len(pop)), key=lambda x: (rank[x], -crowding[x]))
            pop = [pop[i] for i in sorted_idx]

            # now truncate the population
            pop = pop[:self.pop_size]

            #print gen
            #for i in range(len(pop)):
                #print i, pop[i].f, rank[sorted_idx[i]], crowding[sorted_idx[i]]
            #print '---------'

        return pop

    @staticmethod
    def calc_rank_and_crowding(pop):
        fronts = NonDominatedRank.calc_as_fronts(pop)
        rank = NonDominatedRank.calc_from_fronts(fronts)
        cd = np.zeros(len(pop))
        for front in fronts:
            cd_of_front = calc_crowding_distance([pop[x] for x in front])
            for i, idx_of_individual in enumerate(front):
                cd[idx_of_individual] = cd_of_front[i]

        return rank, cd
