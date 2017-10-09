import random

import numpy as np

from model.algorithm import Algorithm
from model.individual import Individual
from operators.polynomial_mutation import PolynomialMutation
from operators.random_factory import RandomFactory
from operators.simulated_binary_crossover import SimulatedBinaryCrossover
from rand.default_random_generator import DefaultRandomGenerator
from util.dominator import Dominator
from util.misc import evaluate
from util.rank_and_crowding import RankAndCrowdingSurvival


class NSGA(Algorithm):
    def __init__(self,
                 pop_size=100,  # population size
                 factory=RandomFactory(),  # factory for the initiation population
                 crossover=SimulatedBinaryCrossover(),  # crossover to be used
                 mutation=PolynomialMutation(),  # mutation to be used
                 survival=RankAndCrowdingSurvival(),  # determine which individuals survive,
                 ):
        self.pop_size = pop_size
        self.factory = factory
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival
        self.rnd = None

    def solve_(self, problem, evaluator, rnd=DefaultRandomGenerator()):

        self.rnd = rnd

        # create the population according to the factoring strategy
        pop = [Individual(x) for x in self.factory.sample(self.pop_size, problem.xl, problem.xu, rnd=self.rnd)]
        evaluate(evaluator, problem, pop)
        rank, crowding = RankAndCrowdingSurvival.calc_rank_and_crowding(pop, rnd=self.rnd)

        # for each generation
        n_gen = 0
        # print 'gen = %d' % (n_gen + 1)

        # while there are functions evaluations left
        while evaluator.has_next():

            n_gen += 1
            # print 'gen = %d' % (n_gen + 1)
            evaluator.notify(pop)

            # create the offspring generation
            offsprings = self.get_offsprings(problem, evaluator, pop, rank, crowding)

            # merge the population and offsprings
            pop = pop + offsprings

            # keep only the individuals that survive
            pop, rank, crowding = self.survival.survive(pop, self.pop_size, rnd=self.rnd)

        evaluator.notify(pop)

        return pop

    def get_offsprings(self, problem, evaluator, pop, rank, crowding):

        perms = self.rnd.permutation(self.pop_size, n=2)

        # do recombination and save offsprings
        offsprings = []
        for i in range(0, self.pop_size, 4):
            # find the parents by doing tournament selection
            parent1 = pop[self.tournament(perms[0][i], perms[0][i + 1], pop, crowding)]
            parent2 = pop[self.tournament(perms[0][i + 2], perms[0][i + 3], pop, crowding)]
            offsprings.extend(self.crossover.crossover(parent1.x, parent2.x, problem.xl, problem.xu, rnd=self.rnd))

            parent1 = pop[self.tournament(perms[1][i], perms[1][i + 1], pop, crowding)]
            parent2 = pop[self.tournament(perms[1][i + 2], perms[1][i + 3], pop, crowding)]
            offsprings.extend(self.crossover.crossover(parent1.x, parent2.x, problem.xl, problem.xu, rnd=self.rnd))

        offsprings = [self.mutation.mutate(off, problem.xl, problem.xu, rnd=self.rnd) for off in offsprings]
        offsprings = [Individual(off) for off in offsprings]

        # evaluate the offspring population
        evaluate(evaluator, problem, offsprings)

        return offsprings

    def tournament(self, i, j, pop, crowding):
        rel = Dominator.get_relation(pop[i], pop[j])
        if rel == 1:
            return i
        elif rel == -1:
            return j
        else:
            if crowding[i] > crowding[j]:
                return i
            elif crowding[i] < crowding[j]:
                return j
            else:
                if self.rnd.random() <= 0.5:
                    return i
                else:
                    return j
