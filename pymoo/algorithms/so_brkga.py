import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.duplicate import DefaultDuplicateElimination, DuplicateElimination
from pymoo.model.population import Population
from pymoo.model.selection import Selection
from pymoo.model.survival import Survival
from pymoo.operators.crossover.biased_crossover import BiasedCrossover
from pymoo.operators.mutation.no_mutation import NoMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Implementation
# =========================================================================================================


class EliteSurvival(Survival):

    def __init__(self, n_elites, eliminate_duplicates=None):
        super().__init__(False)
        self.n_elites = n_elites
        self.eliminate_duplicates = eliminate_duplicates

    def _do(self, problem, pop, n_survive, algorithm=None, **kwargs):

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
            pop = FitnessSurvival().do(problem, pop, len(pop))
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

    def _do(self, pop, n_select, n_parents, **kwargs):
        _type = pop.get("type")
        elites = np.where(_type == "elite")[0]
        non_elites = np.where(_type == "non_elite")[0]

        # if through duplicate elimination no non-elites exist
        if len(non_elites) == 0:
            non_elites = elites

        # do the mating selection - always one elite and one non-elites
        s_elite = elites[RandomSelection().do(elites, n_select, 1)[:, 0]]
        s_non_elite = non_elites[RandomSelection().do(non_elites, n_select, 1)[:, 0]]

        return np.column_stack([s_elite, s_non_elite])


class BRKGA(GeneticAlgorithm):

    def __init__(self,
                 n_elites=200,
                 n_offsprings=700,
                 n_mutants=100,
                 bias=0.7,
                 sampling=FloatRandomSampling(),
                 survival=None,
                 display=SingleObjectiveDisplay(),
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
                         crossover=BiasedCrossover(bias, prob=1.0),
                         mutation=NoMutation(),
                         survival=survival,
                         display=display,
                         eliminate_duplicates=True,
                         **kwargs)

        self.n_elites = n_elites
        self.n_mutants = n_mutants
        self.bias = bias
        self.default_termination = SingleObjectiveDefaultTermination()

    def _next(self):
        pop = self.pop
        elites = np.where(pop.get("type") == "elite")[0]

        # actually do the mating given the elite selection and biased crossover
        off = self.mating.do(self.problem, pop, n_offsprings=self.n_offsprings, algorithm=self)

        # create the mutants randomly to fill the population with
        mutants = FloatRandomSampling().do(self.problem, self.n_mutants, algorithm=self)

        # evaluate all the new solutions
        to_evaluate = Population.merge(off, mutants)
        self.evaluator.eval(self.problem, to_evaluate, algorithm=self)

        # finally merge everything together and sort by fitness
        pop = Population.merge(pop[elites], to_evaluate)

        # the do survival selection - set the elites for the next round
        self.pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)


parse_doc_string(BRKGA.__init__)
