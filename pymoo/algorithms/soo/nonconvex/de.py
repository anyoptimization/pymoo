import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.core.replacement import ImprovementReplacement
from pymoo.core.selection import Selection
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Implementation
# =========================================================================================================


class DE(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 n_offsprings=None,
                 sampling=LHS(),
                 variant="DE/best/1/bin",
                 CR=0.5,
                 F=None,
                 dither="vector",
                 jitter=False,
                 mutation=NoMutation(),
                 survival=None,
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
         The F to be used during the crossover.

        CR : float
         The probability the individual exchanges variable values from the donor vector.

        dither : {{'no', 'scalar', 'vector'}}
         One strategy to introduce adaptive weights (F) during one run. The option allows
         the same dither to be used in one iteration ('scalar') or a different one for
         each individual ('vector).

        jitter : bool
         Another strategy for adaptive weights (F). Here, only a very small value is added or
         subtracted to the F used for the crossover for each individual.

        """

        # parse the information from the string
        _, sel, n_diff, mut, = variant.split("/")
        n_diffs = int(n_diff)
        if "-to-" in variant:
            n_diffs += 1

        selection = DES(sel)

        crossover = DEX(prob=1.0,
                        n_diffs=n_diffs,
                        F=F,
                        CR=CR,
                        variant=mut,
                        dither=dither,
                        jitter=jitter)

        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         display=display,
                         **kwargs)

        self.default_termination = SingleObjectiveDefaultTermination()

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = FitnessSurvival().do(self.problem, infills, n_survive=len(infills))

    def _infill(self):
        infills = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        # tag each individual with an index - if a steady state version is executed
        infills.set("index", np.arange(len(infills)))

        # if number of offsprings is set lower than pop_size - randomly select
        if self.n_offsprings < self.pop_size:
            I = np.random.permutation(len(infills))[:self.n_offsprings]
            infills = infills[I]

        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        # get the indices where each offspring is originating from
        I = infills.get("index")

        # replace the individuals with the corresponding parents from the mating
        self.pop[I] = ImprovementReplacement().do(self.problem, self.pop[I], infills)

        # sort the population by fitness to make the selection simpler for mating (not an actual survival, just sorting)
        self.pop = FitnessSurvival().do(self.problem, self.pop)


# =========================================================================================================
# Selection and Mating
# =========================================================================================================

class DES(Selection):

    def __init__(self, variant) -> None:
        super().__init__()
        self.variant = variant

    def _do(self, pop, n_select, n_parents, **kwargs):
        variant = self.variant

        # create offsprings and add it to the data of the algorithm
        P = RandomSelection().do(pop, n_select, n_parents)

        if variant == "best":
            P[:, 0] = 0
        elif variant == "current-to-best":
            P[:, 0] = np.arange(len(pop))
            P[:, 1] = 0
            P[:, 2] = np.arange(len(pop))
        elif variant == "current-to-rand":
            P[:, 0] = np.arange(len(pop))
            P[:, 2] = np.arange(len(pop))
        elif variant == "rand-to-best":
            P[:, 1] = 0
            P[:, 2] = np.arange(len(pop))
        elif variant == "current-to-pbest":
            # best 10% of the population
            n_pbest = int(np.ceil(0.1 * len(pop)))

            # the corresponding indices to select from
            pbest = np.arange(n_pbest)

            P[:, 0] = np.arange(len(pop))
            P[:, 1] = np.random.choice(pbest, len(pop))
            P[:, 2] = np.arange(len(pop))

        return P


parse_doc_string(DE.__init__)
