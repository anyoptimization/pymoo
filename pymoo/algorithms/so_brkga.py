import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.duplicate import ElementwiseDuplicateElimination, DefaultDuplicateElimination
from pymoo.model.selection import Selection
from pymoo.model.survival import Survival
from pymoo.model.termination import SingleObjectiveToleranceBasedTermination
from pymoo.operators.crossover.biased_crossover import BiasedCrossover
from pymoo.operators.mutation.no_mutation import NoMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================


class PhenotypeDuplicateElimination(ElementwiseDuplicateElimination):

    def __init__(self, cmp=None, **kwargs) -> None:
        super().__init__(cmp, **kwargs)

    def compare(self, a, b):
        pheno_a, pheno_b = a.get("pheno"), b.get("pheno")
        if pheno_a is not None and pheno_b is not None:
            eq = pheno_a == pheno_b
            if isinstance(eq, np.ndarray):
                eq = np.all(eq)
            return 0.0 if eq else 1.0
        else:
            return 1.0


class EliteSurvival(Survival):

    def __init__(self, n_elites, eliminate_duplicates=None) -> None:
        super().__init__(False)
        self.n_elites = n_elites
        self.eliminate_duplicates = eliminate_duplicates

    def _do(self, problem, pop, n_survive, **kwargs):

        if isinstance(self.eliminate_duplicates, bool) and self.eliminate_duplicates:
            pop = DefaultDuplicateElimination(func=lambda p: p.get("F")).do(pop)

        elif isinstance(self.eliminate_duplicates, PhenotypeDuplicateElimination):
            _, _, candidates = DefaultDuplicateElimination(func=lambda pop: pop.get("F")).do(pop, return_indices=True)
            _, _, is_duplicate = self.eliminate_duplicates.do(pop[candidates], return_indices=True)
            elim = candidates[is_duplicate]
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

        # do the mating selection - always one elite and one non-elites
        s_elite = elites[RandomSelection().do(elites, n_select, 1)[:, 0]]
        s_non_elite = non_elites[RandomSelection().do(non_elites, n_select, 1)[:, 0]]

        return np.column_stack([s_elite, s_non_elite])


class BRKGA(GeneticAlgorithm):

    def __init__(self,
                 n_elites=20,
                 n_offsprings=70,
                 n_mutants=10,
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
            Population size

        n_offsprings : int
            Fraction of elite items into each population

        n_mutants : int
            Fraction of mutants introduced at each generation into the population

        bias : float
            Probability that an offspring inherits the allele of its elite parent

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
        self.default_termination = SingleObjectiveToleranceBasedTermination()

    def _next(self):
        pop = self.pop
        elites = np.where(pop.get("type") == "elite")[0]

        # actually do the mating given the elite selection and biased crossover
        off = self._mating(pop)

        # create the mutants randomly to fill the population with
        mutants = FloatRandomSampling().do(self.problem, self.n_mutants, algorithm=self)

        # evaluate all the new solutions
        to_evaluate = off.merge(mutants)
        self.evaluator.eval(self.problem, to_evaluate, algorithm=self)

        # finally merge everything together and sort by fitness
        pop = pop[elites].merge(to_evaluate)

        # the do survival selection - set the elites for the next round
        self.pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)


parse_doc_string(BRKGA.__init__)
