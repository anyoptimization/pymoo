import numpy as np

from algorithms.nsga.rank_and_crowding import RankAndCrowdingSurvival
from model.algorithm import Algorithm
from model.individual import Individual
from operators.polynomial_mutation import PolynomialMutation
from operators.random_factory import RandomFactory
from operators.simulated_binary_crossover import SimulatedBinaryCrossover


class NSGA(Algorithm):
    def __init__(self,
                 pop_size=100,  # population size
                 factory=RandomFactory(),  # factory for the initiation population
                 crossover=SimulatedBinaryCrossover(),  # crossover to be used
                 mutation=PolynomialMutation(),  # mutation to be used
                 survival=RankAndCrowdingSurvival()  # determine which individuals survive
                 ):
        self.pop_size = pop_size
        self.factory = factory
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival

    def solve_(self, problem, evaluator):

        # create the population according to the factoring strategy
        pop = [Individual(x) for x in self.factory.sample(self.pop_size, problem.xl, problem.xu)]
        NSGA.evaluate(evaluator, problem, pop)

        pop = self.survival.survive(pop, self.pop_size)

        # for each generation
        n_gen = 0
        print 'gen = %d' % (n_gen + 1)

        # while there are functions evaluations left
        while evaluator.has_next():
            n_gen += 1
            print 'gen = %d' % (n_gen + 1)

            # create the offspring generation
            offsprings = self.get_offsprings(problem, evaluator, pop)

            # merge the population and offsprings
            pop = pop + offsprings

            # keep only the individuals that survive
            pop = self.survival.survive(pop, self.pop_size)

        return pop

    def get_offsprings(self, problem, evaluator, pop):

        # create the mating pool
        perm = np.concatenate([np.random.permutation(self.pop_size), np.random.permutation(self.pop_size)])

        # do recombination and save offsprings
        offsprings = []
        for i in range(0, len(perm), 4):
            # find the parents by doing tournament selection
            parent1 = pop[min(perm[i], perm[i + 1])]
            parent2 = pop[min(perm[i + 2], perm[i + 3])]

            # do the crossover and mutation
            child1, child2 = self.crossover.crossover(parent1.x, parent2.x, problem.xl, problem.xu)
            child1 = self.mutation.mutate(child1, problem.xl, problem.xu)
            child2 = self.mutation.mutate(child2, problem.xl, problem.xu)

            offsprings.append(Individual(child1))
            offsprings.append(Individual(child2))

        # evaluate the offspring population
        NSGA.evaluate(evaluator, problem, offsprings)

        return offsprings

    @staticmethod
    def evaluate(evaluator, problem, pop):
        for ind in pop:
            ind.f, ind.g = evaluator.eval(problem, ind.x)

    @staticmethod
    def print_pop(pop, rank, crowding, sorted_idx):
        for i in range(len(pop)):
            print i, pop[i].f, rank[sorted_idx[i]], crowding[sorted_idx[i]]
        print '---------'
