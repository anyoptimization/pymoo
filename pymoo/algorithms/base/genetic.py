from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.model.initialization import Initialization
from pymoo.model.mating import Mating
from pymoo.model.population import Population
from pymoo.model.repair import NoRepair


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
                 mating=None,
                 min_infeas_pop_size=0,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # the population size used
        self.pop_size = pop_size

        # minimum number of individuals surviving despite being infeasible - by default disabled
        self.min_infeas_pop_size = min_infeas_pop_size

        # the survival for the genetic algorithm
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        if mating is None:
            mating = Mating(selection,
                            crossover,
                            mutation,
                            repair=self.repair,
                            eliminate_duplicates=self.eliminate_duplicates,
                            n_max_iterations=100)
        self.mating = mating

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)
        return pop

    def _infill(self):

        # do the mating using the current population
        off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, infills)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self,
                                    n_min_infeas_survive=self.min_infeas_pop_size)
