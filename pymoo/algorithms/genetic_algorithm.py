import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population


class GeneticAlgorithm(Algorithm):

    def __init__(self,
                 pop_size,  # population size
                 sampling,  # factory for the initiation population
                 selection,  # methods to selected individuals for mating
                 crossover,  # crossover to be used
                 mutation,  # mutation to be used
                 survival,  # determine what individuals survive
                 verbose=False,
                 plot=False,
                 callback=None
                 ):

        self.pop_size = pop_size
        self.sampling = sampling
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival
        self.verbose = verbose
        self.plot = plot
        self.callback = callback

    def solve_(self, problem, evaluator):

        # create the population according to the factoring strategy
        pop = Population()
        pop.X = self.sampling.sample(problem, self.pop_size)
        pop.F, pop.G = evaluator.eval(problem, pop.X)

        # setup initial generation
        n_gen = 0

        # while there are functions evaluations left
        while evaluator.has_next():

            # increase the generation and do printing and callback
            self.do_each_generation(n_gen, pop)
            n_gen += 1

            # create the offspring generation
            offsprings = self.get_offsprings(problem, evaluator, pop)

            # merge the population and offsprings and do survival selection
            pop.merge(offsprings)
            pop = self.survival.do(pop, self.pop_size)

        self.do_each_generation(n_gen, pop)

        return pop.X, pop.F, pop.G

    def get_offsprings(self, problem, evaluator, pop):

        # initialize selection and offspring methods
        off = Population()
        off.X = np.zeros((self.pop_size, problem.n_var))
        self.selection.initialize(pop)

        n_off = 0
        n_parents = self.crossover.n_parents
        n_children = self.crossover.n_children

        while n_off < self.pop_size:
            # do the mating
            parents = self.selection.next(n_parents)
            X = self.crossover.do(problem, pop.X[parents,:])
            off.X[n_off:min(n_off + n_children, self.pop_size)] = X
            n_off = n_off + len(X)

        off.X = self.mutation.do(problem, off.X)
        off.F, off.G = evaluator.eval(problem, off.X)

        return off

    def do_each_generation(self, n_gen, pop):
        if self.verbose > 0:
            print('gen = %d' % (n_gen + 1))
        if self.verbose > 1:
            pass
        if self.callback is not None:
            self.callback(self, pop)
