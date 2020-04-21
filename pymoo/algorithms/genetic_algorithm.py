from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination
from pymoo.model.individual import Individual
from pymoo.model.initialization import Initialization
from pymoo.model.mating import Mating


class GeneticAlgorithm(Algorithm):

    def __init__(self,
                 pop_size=None,
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 individual=Individual(),
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # the population size used
        self.pop_size = pop_size

        # the survival for the genetic algorithm
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # the object to be used to represent an individual - either individual or derived class
        self.individual = individual

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = None
        else:
            self.eliminate_duplicates = eliminate_duplicates

        self.initialization = Initialization(sampling,
                                             individual=individual,
                                             repair=repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        self.mating = Mating(selection,
                             crossover,
                             mutation,
                             repair=repair,
                             eliminate_duplicates=self.eliminate_duplicates,
                             n_max_iterations=100)

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize(self):

        # create the initial population
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)

        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)

        self.pop, self.off = pop, pop

    def _next(self):

        # do the mating using the current population
        self.off = self.mating.do(self.problem, self.pop, n_offsprings=self.n_offsprings, algorithm=self)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = self.pop.merge(self.off)

        # the do survival selection
        if self.survival:
            self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self)

    def _finalize(self):
        pass
