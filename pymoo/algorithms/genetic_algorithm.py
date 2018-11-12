import math

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.model.algorithm import Algorithm
from pymoo.model.evaluator import Evaluator
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.rand import random


class GeneticAlgorithm(Algorithm):
    """
    This class represents a basic genetic algorithm that can be extended and modified by
    providing different modules or operators.

    Attributes
    ----------

    pop_size: int
        The population size for the genetic algorithm. Depending on the problem complexity and modality the
        it makes sense to experiments with the population size.
        Also, to create a steady state algorithm the offspring_size can be changed.

    sampling : class or numpy.array
        The sampling methodology that is used to create the initial population in the first generation. Also,
        the initial population can be provided directly in case it is known deterministically beforehand.

    selection : model.selection.Selection
        The mating selection methodology that is used to determine the parents for the mating process.

    crossover : model.selection.Crossover
        The crossover methodology that recombines at least two parents to at least one offspring. Depending on
        the arity the number of crossover execution might vary.

    mutation : model.selection.Mutation
        The mutation methodology that is used to perturbate an individual. After performing the crossover
        a mutation is executed.

    survival : model.selection.Survival
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
                 eliminate_duplicates=None,
                 func_repair=None,
                 individual=Individual(),
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # population size of the genetic algorithm
        self.pop_size = pop_size

        # initial sampling method: object, 2d array, or population (already evaluated)
        self.sampling = sampling

        # the method to be used to select parents for recombination
        self.selection = selection

        # method to do the crossover
        self.crossover = crossover

        # method for doing the mutation
        self.mutation = mutation

        # function to repair an offspring after mutation if necessary
        self.func_repair = func_repair

        # survival selection
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # a function that returns the indices of duplicates
        self.eliminate_duplicates = eliminate_duplicates

        # the object to be used to represent an individual - either individual or derived class
        self.individual = individual

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # all run specific data are stored in this dictionary - (is sharing in all modules, e.g. sampling, mutation,..)
        self.data = {}

        # other run specific data updated whenever solve is called - to share them in all methods
        self.evaluator = None
        self.problem = None

    def _solve(self, problem, termination):

        # the evaluator object which is counting the evaluations
        self.evaluator = Evaluator()
        self.problem = problem

        # generation counter
        n_gen = 1

        # always create a new function evaluator which is counting the evaluations
        self.data = {**self.data, 'problem': problem, 'evaluator': self.evaluator,
                     'termination': termination, 'n_gen': n_gen}

        # initialize the first population and evaluate it
        pop = self._initialize()
        self._each_iteration(self.data, first=True)

        # while termination criterium not fulfilled
        while termination.do_continue(self.data):
            self.data['n_gen'] += 1

            # do the next iteration
            pop = self._next(pop)
            self.data = {**self.data, 'pop': pop}

            # execute the callback function in the end of each generation
            self._each_iteration(self.data)

        self._finalize()

        return pop

    def _initialize(self):
        # ! get the initial population - different ways are possible

        # provide a whole population object - (individuals might be already evaluated)
        if isinstance(self.sampling, Population):
            pop = self.sampling
        else:
            pop = Population(0, individual=self.individual)
            if isinstance(self.sampling, np.ndarray):
                pop = pop.new("X", self.sampling)
            else:
                pop = self.sampling.sample(self.problem, pop, self.pop_size, D=self.data)

        # add to the data dictionary to be used in all modules - allows access to pop over data object
        self.data = {**self.data, 'pop': pop}

        # in case the initial population was not evaluated
        if np.any(pop.collect(lambda ind: ind.F is None, as_numpy_array=True)):
            self.evaluator.eval(self.problem, pop, D=self.data)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, self.pop_size, D=self.data)

        return pop

    def _next(self, pop):

        # do the mating using the current population
        off = self._mating(pop)

        # evaluate the offspring
        self.evaluator.eval(self.problem, off, D=self.data)

        # add to the data dictionary of algorithms
        self.data = {**self.data, 'off': off}

        # merge the offsprings with the current population
        pop = pop.merge(off)

        # the do survival selection
        pop = self.survival.do(self.problem, pop, self.pop_size, D=self.data)

        return pop

    def _mating(self, pop):

        # the population object to be used
        off = pop.new()

        # mating counter - counts how often the mating needs to be done to fill up n_offsprings
        n_matings = 0

        # iterate until enough offsprings are created
        while len(off) < self.n_offsprings:

            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil((self.n_offsprings - len(off)) / self.crossover.n_offsprings)

            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, D=self.data)

            # do the crossover using the parents index and the population - additional data provided if necessary
            _off = self.crossover.do(self.problem, pop, parents, D=self.data)

            # do the mutation on the offsprings created through crossover
            _off = self.mutation.do(self.problem, _off, D=self.data)

            # repair the individuals if necessary
            if self.func_repair is not None:
                self.func_repair(self.problem, _off, D=self.data)

            if self.eliminate_duplicates:
                # eliminate duplicates if too close to the current population or already recombined individuals
                if isinstance(self.eliminate_duplicates, bool):
                    is_duplicate = default_is_duplicate(_off, pop, off)
                elif isinstance(self.eliminate_duplicates, function):
                    is_duplicate = self.eliminate_duplicates(_off, pop, off)
                # filter out the duplicates from the offsprings before adding
                _off = _off[np.logical_not(is_duplicate)]

            # if more offsprings than necessary - truncate them
            if len(_off) > self.n_offsprings - len(off):
                I = random.perm(self.n_offsprings - len(off))
                _off = _off[I]

            # add to the offsprings and increase the mating counter
            off = off.merge(_off)
            n_matings += 1

            # if no new offsprings can be generated within 100 trails -> return the current result
            if n_matings > 100:
                self.data['termination'].flag = False
                break

        return off

    def _finalize(self):
        pass


def default_is_duplicate(pop, *other, epsilon=1e-20):
    if len(other) == 0:
        return np.full(len(pop), False)

    X = pop.get("X")

    _X = other[0].get("X")
    for o in other[1:]:
        if len(o) > 0:
            _X = np.concatenate([_X, o.get("X")])

    is_duplicate = np.any(cdist(X, _X) < epsilon, axis=1)

    return is_duplicate
