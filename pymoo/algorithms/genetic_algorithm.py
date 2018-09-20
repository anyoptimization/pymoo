import math

import numpy as np

from pymoo.model.individual import Individual
from pymoo.model.individual import get_genome, create_from_genome, create_as_objects
from pymoo.model.algorithm import Algorithm
from pymoo.model.evaluator import Evaluator
from pymoo.model.population import Population
from pymoo.util.misc import cdist


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
                 func_repair=None,
                 clazz=Individual().__class__,
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
        self.func_repair = func_repair
        self.clazz = clazz
        self.D = {}

        # default set the number of offsprings to the population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

    def _next(self, pop):

        # do the iteration for the next generation - population objective is modified inplace
        # do the mating and evaluate the offsprings
        off = self._mating(pop)

        # if it is a custom class also create the custom object and store it in the population
        off.individuals = create_from_genome(self.clazz, off.X)

        off.F, off.CV, off.G = self.D['evaluator'].eval(self.D['problem'], off.X, D=self.D, individuals=off.individuals,
                                                        return_constraint_violation=True, return_constraints=True)
        self.D = {**self.D, 'off': off}

        # do the survival selection with the merged population
        pop.merge(off)
        self.survival.do(pop, self.pop_size, D=self.D)

        return off

    def _solve(self, problem, termination):

        # the evaluator object which is counting the evaluations
        evaluator = Evaluator()

        # generation counter
        n_gen = 1

        # always create a new function evaluator which is counting the evaluations
        self.D = {**self.D, 'problem': problem, 'evaluator': evaluator, 'n_gen': n_gen}

        # initialize the first population and evaluate it
        pop = self._initialize()
        self.D = {**self.D, 'pop': pop}
        self._each_iteration(self.D, first=True)

        # while termination criterium not fulfilled
        while termination.do_continue(self.D):
            self.D['n_gen'] += 1

            # do the next iteration
            self._next(pop)

            # execute the callback function in the end of each generation
            self._each_iteration(self.D)

        self._finalize()

        return pop

    def _initialize(self):

        problem, evaluator = self.D['problem'], self.D['evaluator']

        pop = Population()

        # add to the data dictionary to be used in all modules
        self.D = {**self.D, 'pop': pop}

        if isinstance(self.sampling, np.ndarray):
            pop.X = self.sampling
        else:
            pop.X = self.sampling.sample(problem, self.pop_size, D=self.D)

        # if we got a custom object during the sampling - figure out the clazz type directly
        if pop.X.dtype == np.object:

            # set the class of the custom object to be used for factory later
            self.clazz = pop.X[0, 0].__class__
            obj = pop.X[0, 0].get_genome()

            # now either they return also float - we extract the matrix
            if obj is not None and obj.dtype == np.float:
                pop.individuals = pop.X
                pop.X = get_genome(pop.individuals)

        # the object has a type which can be represented in a matrix - only if clazz is not None create an extra object
        else:

            # if no individual object is desired create a dummy array
            if self.clazz is None:
                pop.individuals = np.full((self.pop_size, 1), np.inf)

            # otherwise we create objects from this clazz
            else:
                pop.individuals = create_as_objects(self.clazz, self.pop_size)

        # add to the data dictionary to be used in all modules
        self.D = {**self.D, 'pop': pop}

        pop.F, pop.CV, pop.G = evaluator.eval(problem, pop.X, D=self.D, individuals=pop.individuals,
                                              return_constraint_violation=True, return_constraints=True)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        self.survival.do(pop, self.pop_size, D=self.D)

        return pop

    def _mating(self, pop):

        off = Population(X=np.full((0, self.D['problem'].n_var), np.inf), individuals=create_as_objects(self.clazz, 0))
        n_gen_off = 0
        n_parents = self.crossover.n_parents

        # mating counter - counts how often the mating needs to be done to fill up n_offsprings
        n_matings = 0

        # do the mating until all offspring are created usually it should be done is one iteration
        # through duplicate eliminate more might be necessary
        while n_gen_off < self.n_offsprings:

            # select from the current population individuals for mating
            n_select = math.ceil((self.n_offsprings - n_gen_off) / self.crossover.n_children)
            parents = self.selection.do(pop, n_select, n_parents, D=self.D, indviduals=pop.individuals)

            # object that represent the individuals
            _individuals = create_as_objects(self.clazz, parents.shape[0] * self.crossover.n_children)

            _X = self.crossover.do(self.D['problem'], pop.X[parents.T], D=self.D, individuals=_individuals)

            # do the mutation
            _X = self.mutation.do(self.D['problem'], _X, D=self.D, individuals=_individuals)

            # repair the individuals if necessary
            if self.func_repair is not None:
                self.func_repair(self.D['problem'], _X, D=self.D, individuals=_individuals)

            # eliminate duplicates if too close to the current population
            if self.eliminate_duplicates:
                not_equal = np.where(np.all(cdist(_X, pop.X) > 1e-12, axis=1))[0]
                _X, _individuals = _X[not_equal, :], _individuals[not_equal, :]

            # if more offsprings than necessary - truncate them
            if _X.shape[0] > self.n_offsprings - n_gen_off:
                _X = _X[:self.n_offsprings - n_gen_off, :]
                _individuals = _individuals[:self.n_offsprings - n_gen_off, :]

            # add to the offsprings
            off.merge(Population(X=_X, individuals=_individuals))
            n_gen_off += _X.shape[0]

            # increase the mating number
            n_matings += 1

            # if no new offsprings can be generated within 100 trails -> return the current result
            if n_matings > 100:
                self.evaluator.n_eval = self.evaluator.n_max_eval
                break

        return off

    def _finalize(self):
        pass
