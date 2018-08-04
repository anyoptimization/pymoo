import math

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population


class GeneticAlgorithm(Algorithm):
    """
    This class represents a basic genetic algorithm that can be extended and modified by
    providing different modules or operators.

    Attributes
    ----------

    pop_size: int
        The population size for the genetic algorithm. Depending on the problem complexity and modality the
        it makes sense to experiment with the population size.
        Also, to create a steady state algorithm the offspring_size can be changed.

    sampling : class or numpy.array
        The sampling methodology that is used to create the initial population in the first generation. Also,
        the initial population can be provided directly in case it is known deterministically beforehand.

    selection : pymoo.model.selection.Selection
        The mating selection methodology that is used to determine the parents for the mating process.

    crossover : pymoo.model.selection.Crossover
        The crossover methodology that recombines at least two parents to at least one offspring. Depending on
        the arity the number of crossover execution might vary.

    mutation : pymoo.model.selection.Mutation
        The mutation methodology that is used to perturbate an individual. After performing the crossover
        a mutation is executed.

    survival : pymoo.model.selection.Survival
        Each generation usually a survival selection is performed to follow the survival of the fittest principle.
        However, other strategies such as niching, diversity preservation and so on can be implemented here.

    n_offsprings : int
        Number of offsprings to be generated each generation. Can be 1 to define a steady-state algorithm.
        Default it is equal to the population size.

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
        self.n_offsprings = n_offsprings
        self.eliminate_duplicates = eliminate_duplicates

        # default set the number of offsprings to the population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

    def _next(self, pop):

        # do the iteration for the next generation - population objective is modified inplace
        # do the mating and evaluate the offsprings
        off = Population()
        off.X = self._mating(pop)
        off.F, off.G = self.evaluator.eval(self.problem, off.X)

        # do the survival selection with the merged population
        self.survival.do(pop, off, self.pop_size, out=self.D, **self.D)

        return off

    def _solve(self, problem, evaluator):

        # for convenience add to class attributes for access in all sub-methods
        self.problem, self.evaluator = problem, evaluator

        # dictionary that shared among the modules. Note, here variables can be modified and the changes will reflect.
        # all variables here are only valid for one run using the solve method.
        self.D = {}

        # generation counter
        n_gen = 0

        # initialize the first population and evaluate it
        pop = self._initialize()
        self._each_iteration({'algorithm': self, 'n_gen': n_gen, 'pop': pop, **self.D}, first=True)

        # while there are functions evaluations left
        while evaluator.has_remaining():
            n_gen += 1

            # do the next iteration
            self._next(pop)

            # execute the callback function in the end of each generation
            self._each_iteration({'algorithm': self, 'n_gen': n_gen, 'pop': pop, **self.D})

        return pop.X, pop.F, pop.G

    def _initialize(self):

        pop = Population()
        if isinstance(self.sampling, np.ndarray):
            pop.X = self.sampling
        else:
            pop.X = self.sampling.sample(self.problem, self.pop_size)
        pop.F, pop.G = self.evaluator.eval(self.problem, pop.X)
        return pop

    def _mating(self, pop):

        # initialize selection and offspring methods
        X = np.full((self.n_offsprings, self.problem.n_var), np.inf)
        n_offsprings = 0
        n_parents = self.crossover.n_parents

        # mating counter - counts how often the mating needs to be done to fill up n_offsprings
        n_matings = 0

        # do the mating until all offspring are created usually it should be done is one iteration
        # through duplicate eliminate more might be necessary
        while n_offsprings < self.n_offsprings:

            # select from the current population individuals for mating
            n_select = math.ceil((self.n_offsprings - n_offsprings) / self.crossover.n_children)
            parents = self.selection.do(pop, n_select, n_parents, out=self.D, **self.D)
            X = self.crossover.do(self.problem, pop.X[parents, :], out=self.D, **self.D)

            # do the mutation
            X = self.mutation.do(self.problem, X, out=self.D, **self.D)

            # if more offsprings than necessary - truncate them
            if X.shape[0] > self.n_offsprings - n_offsprings:
                X = X[:self.n_offsprings - n_offsprings, :]

            # eliminate duplicates if too close to the current population
            if self.eliminate_duplicates:
                is_equal = np.min(cdist(X, pop.X), axis=1) <= 1e-60
                X = X[np.logical_not(is_equal), :]

            # add to the offsprings
            X[n_offsprings:n_offsprings + X.shape[0], :] = X
            n_offsprings = n_offsprings + X.shape[0]

            n_matings += 1

            # if no new offsprings can be generated within 100 trails -> return the current result
            if n_matings > 100:
                X = X[:n_offsprings, :]
                self.evaluator.n_eval = self.evaluator.n_max_eval
                break

        return X
