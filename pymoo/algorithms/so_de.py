import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement
from pymoo.operators.crossover.biased_crossover import BiasedCrossover
from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.crossover.exponential_crossover import ExponentialCrossover
from pymoo.operators.repair.bounce_back import BounceBackOutOfBoundsRepair
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import parameter_less
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Implementation
# =========================================================================================================


class DE(GeneticAlgorithm):
    def __init__(self,
                 pop_size=100,
                 sampling=LatinHypercubeSampling(),
                 crossover=None,
                 variant="DE/rand/1/bin",
                 CR=0.5,
                 F=0.3,
                 dither="vector",
                 jitter=False,
                 display=SingleObjectiveDisplay(),
                 **kwargs
                 ):

        """

        Parameters
        ----------

        pop_size : {pop_size}

        sampling : {sampling}

        variant : {{DE/(rand|best)/1/(bin/exp)}}
         The different variants of DE to be used. DE/x/y/z where x how to select individuals to be pertubed,
         y the number of difference vector to be used and z the crossover type. One of the most common variant
         is DE/rand/1/bin.

        F : float
         The weight to be used during the crossover.

        CR : float
         The probability the individual exchanges variable values from the donor vector.

        dither : {{'no', 'scalar', 'vector'}}
         One strategy to introduce adaptive weights (F) during one run. The option allows
         the same dither to be used in one iteration ('scalar') or a different one for
         each individual ('vector).

        jitter : bool
         Another strategy for adaptive weights (F). Here, only a very small value is added or
         subtracted to the weight used for the crossover for each individual.


        """

        _, self.var_selection, self.var_n, self.var_mutation, = variant.split("/")

        if self.var_mutation == "exp":
            mutation = ExponentialCrossover(CR)
        elif self.var_mutation == "bin":
            mutation = BiasedCrossover(CR)

        if crossover is None:
            crossover = DifferentialEvolutionCrossover(weight=F, dither=dither, jitter=jitter)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=RandomSelection(),
                         crossover=crossover,
                         mutation=mutation,
                         survival=None,
                         display=display,
                         **kwargs)

        self.default_termination = SingleObjectiveDefaultTermination()

    def _next(self):

        # make a step and create the offsprings
        self.off = self._step()

        # evaluate the offsprings
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # replace the individuals that have improved
        self.pop = ImprovementReplacement().do(self.problem, self.pop, self.off)

    def _step(self):
        selection, crossover, mutation = self.mating.selection, self.mating.crossover, self.mating.mutation

        # retrieve the current population
        pop = self.pop

        # get the vectors from the population
        F, CV, feasible = pop.get("F", "CV", "feasible")
        F = parameter_less(F, CV)

        # create offsprings and add it to the data of the algorithm
        P = selection.do(pop, self.pop_size, crossover.n_parents)

        if self.var_selection == "best":
            P[:, 0] = np.argmin(F[:, 0])
        elif self.var_selection == "rand+best":
            P[np.random.random(len(pop)) < 0.3, 0] = np.argmin(F[:, 0])

        # do the first crossover which is the actual DE operation
        off = crossover.do(self.problem, pop, P, algorithm=self)

        # then do the mutation (which is actually a crossover between old and new individual)
        _pop = Population.merge(self.pop, off)
        _P = np.column_stack([np.arange(len(pop)), np.arange(len(pop)) + len(pop)])
        off = mutation.do(self.problem, _pop, _P, algorithm=self)[:len(self.pop)]

        # bounds back if something is out of bounds
        off = BounceBackOutOfBoundsRepair().do(self.problem, off)

        return off


parse_doc_string(DE.__init__)
