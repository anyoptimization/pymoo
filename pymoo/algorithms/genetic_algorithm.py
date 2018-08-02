import math

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population


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
        The sampling strategy to create the initial population.

    selection : class
        The method to select the parents for the crossover

    crossover : class
        The crossover to be performed on parents.

    mutation : class
        The mutation that will be performed for each child after the crossover

    survival : class
        This class selects the individuals to survive for the next generation

    n_offsprings : int
        Number of offspring to be generate each epoch. Can be 1 to create a steady-state algorithm.

    eliminate_duplicates : bool
        If this flag is set no duplicates are allowed in the population (mostly likely only used for binary or discrete)

    """

    def __init__(self,
                 pop_size,
                 sampling,
                 selection,
                 crossover,
                 mutation,
                 survival,
                 n_offsprings=None,
                 eliminate_duplicates=False,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.sampling = sampling
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival
        self.eliminate_duplicates = eliminate_duplicates
        self.n_offsprings = n_offsprings

        # dictionary which is used to share data among modules
        self.data = {}

        # default set the number of offsprings to the population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

    def _solve(self, problem, evaluator):

        # setup initial generation
        n_gen = 0

        # create the population according to the factoring strategy
        pop = Population()
        if isinstance(self.sampling, np.ndarray):
            pop.X = self.sampling
        else:
            pop.X = self.sampling.sample(problem, self.pop_size)
        pop.F, pop.G = evaluator.eval(problem, pop.X)
        pop = self.survival.do(pop, self.pop_size, data=self.data)

        # while there are functions evaluations left
        while evaluator.has_remaining():

            # increase the generation and do printing and callback
            self._do_each_iteration(n_gen, evaluator, pop)
            n_gen += 1

            # initialize selection and offspring methods
            off = Population()
            off.X = np.full((self.n_offsprings, problem.n_var), np.inf)
            n_offsprings = 0
            n_parents = self.crossover.n_parents

            # mating counter - counts how often the mating needs to be done to fill up n_offsprings
            n_matings = 0

            # do the mating until all offspring are created usually it should be done is one iteration
            # through duplicate eliminate more might be necessary
            while n_offsprings < self.n_offsprings:

                # select from the current population individuals for mating
                n_select = math.ceil((self.n_offsprings - n_offsprings) / self.crossover.n_children)
                parents = self.selection.next(pop, n_select, n_parents, data=self.data)
                X = self.crossover.do(problem, pop.X[parents, :])

                # if more offsprings than necessary - truncate them
                if X.shape[0] > self.n_offsprings - n_offsprings:
                    X = X[:self.n_offsprings - n_offsprings, :]

                # do the mutation
                X = self.mutation.do(problem, X)

                # eliminate duplicates if too close to the current population
                if self.eliminate_duplicates:
                    is_equal = np.min(cdist(X, pop.X), axis=1) <= 1e-60
                    X = X[np.logical_not(is_equal), :]

                # add to the offsprings
                off.X[n_offsprings:n_offsprings + X.shape[0], :] = X
                n_offsprings = n_offsprings + X.shape[0]

                n_matings += 1

                # if no new offsprings can be generated within 100 trails -> return the current result
                if n_matings > 100:
                    off.X = off.X[:n_offsprings, :]
                    evaluator.n_eval = evaluator.n_max_eval
                    break

            off.F, off.G = evaluator.eval(problem, off.X)

            # merge the population
            pop.merge(off)

            # truncate the population
            pop = self.survival.do(pop, self.pop_size)

        self._do_each_iteration(n_gen, evaluator, pop)
        return pop.X, pop.F, pop.G

    def _do_each_iteration(self, n_iteration, evaluator, pop):
        if self.verbose > 0:
            print('gen = %d (FE:%s)' % (n_iteration + 1, evaluator.n_eval))
        if self.verbose > 1:
            pass
        if self.callback is not None:
            self.callback(self, evaluator.counter, pop)

        if self.history is not None:
            self.history.append(
                {'n_gen': n_iteration,
                 'n_evals': evaluator.n_eval,
                 'X': np.copy(pop.X),
                 'F': np.copy(pop.F)
                 })
