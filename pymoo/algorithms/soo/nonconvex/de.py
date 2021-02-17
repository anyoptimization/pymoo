import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.mating import Mating
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement
from pymoo.model.selection import Selection
from pymoo.operators.crossover.biased_crossover import BiasedCrossover
from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.crossover.exponential_crossover import ExponentialCrossover
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
                 n_offsprings=100,
                 sampling=LatinHypercubeSampling(),
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

        mating = DifferentialEvolutionMating(variant=variant,
                                             CR=CR,
                                             F=F,
                                             dither=dither,
                                             jitter=jitter)

        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         mating=mating,
                         survival=None,
                         display=display,
                         **kwargs)

        self.default_termination = SingleObjectiveDefaultTermination()

    def _infill(self):
        infills = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        if len(self.pop) != len(infills):
            self.indices = np.random.permutation(len(self.pop))[:len(infills)]
        else:
            self.indices = np.arange(len(self.pop))

        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."
        self.pop[self.indices] = ImprovementReplacement().do(self.problem, self.pop[self.indices], infills)


# =========================================================================================================
# Selection and Mating
# =========================================================================================================

class DESelection(Selection):

    def __init__(self, variant) -> None:
        super().__init__()
        self.variant = variant

    def _do(self, pop, n_select, n_parents, **kwargs):
        variant = self.variant

        # create offsprings and add it to the data of the algorithm
        P = RandomSelection().do(pop, n_select, n_parents)

        F, CV = pop.get("F", "CV")
        fitness = parameter_less(F, CV)[:, 0]
        sorted_by_fitness = fitness.argsort()
        best = sorted_by_fitness[0]

        if variant == "best":
            P[:, 0] = best
        elif variant == "current-to-best":
            P[:, 0] = np.arange(len(pop))
            P[:, 1] = best
            P[:, 2] = np.arange(len(pop))
        elif variant == "current-to-rand":
            P[:, 0] = np.arange(len(pop))
            P[:, 2] = np.arange(len(pop))
        elif variant == "rand-to-best":
            P[:, 1] = best
            P[:, 2] = np.arange(len(pop))
        elif variant == "current-to-pbest":
            n_pbest = int(np.ceil(0.1 * len(pop)))
            pbest = sorted_by_fitness[:n_pbest]

            P[:, 0] = np.arange(len(pop))
            P[:, 1] = np.random.choice(pbest, len(pop))
            P[:, 2] = np.arange(len(pop))

        return P


class DifferentialEvolutionMating(Mating):

    def __init__(self,
                 variant="DE/rand/1/bin",
                 CR=0.5,
                 F=0.3,
                 dither="vector",
                 jitter=False,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 **kwargs):

        _, sel, n_diff, mut, = variant.split("/")
        self.variant = sel
        self.n_diffs = int(n_diff)
        if "-to-" in self.variant:
            self.n_diffs += 1

        if selection is None:
            selection = DESelection(sel)

        if mutation is None:
            if mut == "exp":
                mutation = ExponentialCrossover(CR)
            elif mut == "bin":
                mutation = BiasedCrossover(CR)

        if crossover is None:
            crossover = DifferentialEvolutionCrossover(n_diffs=self.n_diffs, weight=F, dither=dither, jitter=jitter)

        super().__init__(selection, crossover, mutation, **kwargs)

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        P = self.selection.do(pop, len(pop), self.crossover.n_parents)

        # do the first crossover which is the actual DE operation
        off = self.crossover.do(problem, pop, P, algorithm=self)

        # then do the mutation (which is actually a crossover between old and new individual)
        _pop = Population.merge(pop, off)
        _P = np.column_stack([np.arange(len(pop)), np.arange(len(pop)) + len(pop)])
        off = self.mutation.do(problem, _pop, _P, algorithm=self)[:len(pop)]

        return off


parse_doc_string(DE.__init__)
