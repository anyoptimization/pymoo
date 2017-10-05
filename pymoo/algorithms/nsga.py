import random

import numpy as np

from model.algorithm import Algorithm
from model.individual import Individual
from operators.polynomial_mutation import PolynomialMutation
from operators.random_factory import RandomFactory
from operators.simulated_binary_crossover import SimulatedBinaryCrossover
from rand.default_random_generator import DefaultRandomGenerator
from util.misc import evaluate
from util.rank_and_crowding import RankAndCrowdingSurvival


class NSGA(Algorithm):
    def __init__(self,
                 pop_size=100,  # population size
                 factory=RandomFactory(),  # factory for the initiation population
                 crossover=SimulatedBinaryCrossover(),  # crossover to be used
                 mutation=PolynomialMutation(),  # mutation to be used
                 survival=RankAndCrowdingSurvival(),  # determine which individuals survive,
                 rnd=DefaultRandomGenerator()
                 ):
        self.pop_size = pop_size
        self.factory = factory
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival
        self.rnd = rnd

    def solve_(self, problem, evaluator):

        # create the population according to the factoring strategy
        pop = [Individual(x) for x in self.factory.sample(self.pop_size, problem.xl, problem.xu, rnd=self.rnd)]
        evaluate(evaluator, problem, pop)
        pop = self.survival.survive(pop, self.pop_size)

        # for each generation
        n_gen = 0
        # print 'gen = %d' % (n_gen + 1)

        # while there are functions evaluations left
        while evaluator.has_next():
            n_gen += 1
            # print 'gen = %d' % (n_gen + 1)
            evaluator.notify(pop)

            # create the offspring generation
            offsprings = self.get_offsprings(problem, evaluator, pop)

            # merge the population and offsprings
            pop = pop + offsprings

            # keep only the individuals that survive
            random.shuffle(pop)
            pop = self.survival.survive(pop, self.pop_size)

        evaluator.notify(pop)

        return pop

    def get_offsprings(self, problem, evaluator, pop):

        perm = np.array(self.rnd.permutation(self.pop_size))
        perm = np.append(perm, np.array(self.rnd.permutation(self.pop_size)))

        # do recombination and save offsprings
        offsprings = []
        for i in range(0, len(perm), 4):
            # find the parents by doing tournament selection
            parent1 = pop[min(perm[i], perm[i + 1])]
            parent2 = pop[min(perm[i + 2], perm[i + 3])]

            # do the crossover
            children = self.crossover.crossover(parent1.x, parent2.x, problem.xl, problem.xu, rnd=self.rnd)
            # mutate
            children = [self.mutation.mutate(child, problem.xl, problem.xu, rnd=self.rnd) for child in children]

            offsprings.extend([Individual(child) for child in children])

        # evaluate the offspring population
        evaluate(evaluator, problem, offsprings)

        return offsprings
