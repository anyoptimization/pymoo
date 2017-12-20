import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population
from pymoo.util.misc import unique_rows, create_hist


class GeneticAlgorithm(Algorithm):
    """

    This class represents a basic genetic algorithm that can be extended and modified by
    providing different modules in the constructor. Most importantly the modules for the initial sampling,
    recombination and the survival selection should be provided.
    Different implementations of algorithm can use this class.

    Attributes
    ----------
    pop_size: int
        The population size to be used for the genetic algorithm.
    sampling : class
        The sampling implementation to create the initial population.
    selection : class
        A class to select the parents for the crossover
    crossover : class
        The crossover to be performed on parents.
    mutation : class
        The mutation that will be performed for each child after the crossover
    survival : class
        This class selects the individuals to survive for the next generation
    eliminate_duplicates : bool
        If this flag is set no duplicates are allowed in the population (mostly likely only used for binary or discrete)
    verbose : int
        If larger than zero output is provided. (verbose=1 means some output, verbose=2 details for debugging)
    callback : func
        A callback function can be passed that is executed every generation. The parameters for the function
        are the algorithm itself, the number of evaluations so far and the current population.

            def callback(algorithm, n_evals, pop):
                print()

    """

    def __init__(self,
                 pop_size,
                 sampling,
                 selection,
                 crossover,
                 mutation,
                 survival,
                 eliminate_duplicates=False,
                 verbose=False,
                 callback=None
                 ):

        self.pop_size = pop_size
        self.sampling = sampling
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival
        self.eliminate_duplicates = eliminate_duplicates
        self.verbose = verbose
        self.callback = callback

    def _initialize(self):
        pass

    def _solve(self, problem, evaluator):

        # initialize stuff before the initial population is created
        self._initialize()

        # create the population according to the factoring strategy
        pop = Population()
        pop.X = self.sampling.sample(problem, self.pop_size)
        pop.F, pop.G = evaluator.eval(problem, pop.X)
        pop = self.survival.do(pop, self.pop_size)

        # setup initial generation
        n_gen = 0

        # while there are functions evaluations left
        while evaluator.has_next():

            # increase the generation and do printing and callback
            self._do_each_generation(n_gen, evaluator, pop)
            n_gen += 1

            # create the offspring generation
            offsprings = self._get_offsprings(problem, evaluator, pop)

            # merge the population
            pop.merge(offsprings)

            # eliminate all duplicates in the population
            if self.eliminate_duplicates:
                pop.filter(unique_rows(pop.X))

            # truncate the population
            pop = self.survival.do(pop, self.pop_size)

        self._do_each_generation(n_gen, evaluator, pop)

        return pop.X, pop.F, pop.G

    def _get_offsprings(self, problem, evaluator, pop):

        # initialize selection and offspring methods
        off = Population()
        off.X = np.zeros((self.pop_size, problem.n_var))
        self.selection._initialize(pop)

        n_off = 0
        n_parents = self.crossover.n_parents
        n_children = self.crossover.n_children

        while n_off < self.pop_size:
            parents = self.selection.next(n_parents)
            X = self.crossover.do(problem, pop.X[parents, :])

            off.X[n_off:min(n_off + n_children, self.pop_size)] = X
            n_off = n_off + len(X)

        off.X = self.mutation.do(problem, off.X)
        off.F, off.G = evaluator.eval(problem, off.X)

        return off

    def _do_each_generation(self, n_gen, evaluator, pop):
        if self.verbose > 0:
            print('gen = %d' % (n_gen + 1))
        if self.verbose > 1:
            pass
        if self.callback is not None:
            self.callback(self, evaluator.counter, pop)
        evaluator.notify(
            {'n_gen': n_gen,
             'n_evals': evaluator.counter,
             'snapshot': create_hist(evaluator.counter, pop)
             })
