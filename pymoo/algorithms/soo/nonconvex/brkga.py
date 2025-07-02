import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import DefaultDuplicateElimination, DuplicateElimination
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================


class EliteSurvival(Survival):

    def __init__(self, n_elites, eliminate_duplicates=None):
        super().__init__(False)
        self.n_elites = n_elites
        self.eliminate_duplicates = eliminate_duplicates

    def _do(self, problem, pop, n_survive=None, algorithm=None, **kwargs):

        if isinstance(self.eliminate_duplicates, bool) and self.eliminate_duplicates:
            pop = DefaultDuplicateElimination(func=lambda p: p.get("F")).do(pop)

        elif isinstance(self.eliminate_duplicates, DuplicateElimination):
            _, no_candidates, candidates = DefaultDuplicateElimination(func=lambda pop: pop.get("F")).do(pop,
                                                                                                         return_indices=True)
            _, _, is_duplicate = self.eliminate_duplicates.do(pop[candidates], pop[no_candidates], return_indices=True,
                                                              to_itself=False)
            elim = set(np.array(candidates)[is_duplicate])
            pop = pop[[k for k in range(len(pop)) if k not in elim]]

        if problem.n_obj == 1:
            pop = FitnessSurvival().do(problem, pop, n_survive=len(pop))
            elites = pop[:self.n_elites]
            non_elites = pop[self.n_elites:]
        else:
            I = NonDominatedSorting().do(pop.get("F"), only_non_dominated_front=True)
            elites = pop[I]
            non_elites = pop[[k for k in range(len(pop)) if k not in I]]

        elites.set("type", ["elite"] * len(elites))
        non_elites.set("type", ["non_elite"] * len(non_elites))

        return pop


class EliteBiasedSelection(Selection):

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        random_state = kwargs.get('random_state')
        _type = pop.get("type")
        elites = np.where(_type == "elite")[0].astype(int)
        non_elites = np.where(_type == "non_elite")[0].astype(int)

        # if through duplicate elimination no non-elites exist
        if len(non_elites) == 0:
            non_elites = elites

        # do the mating selection - always one elite and one non-elites
        s_elite = random_state.choice(elites, size=n_select)
        s_non_elite = random_state.choice(non_elites, size=n_select)

        return np.column_stack([s_elite, s_non_elite])


class BRKGA(GeneticAlgorithm):

    def __init__(self,
                 n_elites=200,
                 n_offsprings=700,
                 n_mutants=100,
                 bias=0.7,
                 sampling=FloatRandomSampling(),
                 survival=None,
                 output=SingleObjectiveOutput(),
                 eliminate_duplicates=False,
                 **kwargs
                 ):
        """


        Parameters
        ----------

        n_elites : int
            Number of elite individuals

        n_offsprings : int
            Number of offsprings to be generated through mating of an elite and a non-elite individual

        n_mutants : int
            Number of mutations to be introduced each generation

        bias : float
            Bias of an offspring inheriting the allele of its elite parent

        eliminate_duplicates : bool or class
            The duplicate elimination is more important if a decoding is used. The duplicate check has to be
            performed on the decoded variable and not on the real values. Therefore, we recommend passing
            a DuplicateElimination object.
            If eliminate_duplicates is simply set to `True`, then duplicates are filtered out whenever the
            objective values are equal.

        """

        if survival is None:
            survival = EliteSurvival(n_elites, eliminate_duplicates=eliminate_duplicates)

        super().__init__(pop_size=n_elites + n_offsprings + n_mutants,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         selection=EliteBiasedSelection(),
                         crossover=BinomialCrossover(bias, prob=1.0, n_offsprings=1),
                         mutation=NoMutation(),
                         survival=survival,
                         output=output,
                         eliminate_duplicates=True,
                         advance_after_initial_infill=True,
                         **kwargs)

        self.n_elites = n_elites
        self.n_mutants = n_mutants
        self.bias = bias
        self.termination = DefaultSingleObjectiveTermination()

    def _infill(self):
        pop = self.pop

        # actually do the mating given the elite selection and biased crossover
        off = self.mating.do(self.problem, pop, n_offsprings=self.n_offsprings, algorithm=self, random_state=self.random_state)

        # create the mutants randomly to fill the population with
        mutants = FloatRandomSampling().do(self.problem, self.n_mutants, algorithm=self, random_state=self.random_state)

        # evaluate all the new solutions
        return Population.merge(off, mutants)

    def _advance(self, infills=None, **kwargs):
        pop = self.pop

        # get all the elites from the current population
        elites = np.where(pop.get("type") == "elite")[0]

        # finally merge everything together and sort by fitness
        pop = Population.merge(pop[elites], infills)

        # the do survival selection - set the elites for the next round
        self.pop = self.survival.do(self.problem, pop, n_survive=len(pop), algorithm=self)


parse_doc_string(BRKGA.__init__)
