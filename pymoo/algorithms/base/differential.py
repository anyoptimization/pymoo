# External
import numpy as np

from pymoo.core.population import Population
from pymoo.algorithms.base.evolutionary import EvolutionaryAlgorithm
from pymoo.core.infill import InfillCriterion

from pymoo.operators.selection.des import DES
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.crossover.dem import DEM

from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.misc import has_feasible

from pymoo.core.replacement import ImprovementReplacement
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pymoo.termination.default import DefaultSingleObjectiveTermination, DefaultMultiObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.display.multi import MultiObjectiveOutput


# =========================================================================================================
# Differential Evolution Variant
# =========================================================================================================

class DifferentialVariant(InfillCriterion):

    def __init__(self,
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=(0.5, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 genetic_mutation=None,
                 **kwargs):
        """InfillCriterion class for Differential Evolution

        Parameters
        ----------
        variant : str, optional
            Differential evolution strategy. Must be a string in the format: "DE/selection/n/crossover", in which, n in an integer of number of difference vectors, and crossover is either 'bin' or 'exp'. Selection variants are:

                - 'ranked'
                - 'rand'
                - 'best'
                - 'current-to-best'
                - 'current-to-rand'
                - 'rand-to-best'

            The selection strategy 'ranked' might be helpful to improve convergence speed without much harm to diversity. Defaults to 'DE/rand/1/bin'.

        CR : float, optional
            Crossover parameter. Defined in the range [0, 1]
            To reinforce mutation, use higher values. To control convergence speed, use lower values.

        F : iterable of float or float, optional
            Scale factor or mutation parameter. Defined in the range (0, 2]
            To reinforce exploration, use higher values; for exploitation, use lower values.

        gamma : float, optional
            Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.

        de_repair : str, optional
            Repair of DE mutant vectors. Is either callable or one of:

                - 'bounce-back'
                - 'midway'
                - 'rand-init'
                - 'to-bounds'

            If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors including violations, Xb contains reference vectors for repair in feasible space, xl is a 1d vector of lower bounds, and xu a 1d vector of upper bounds.
            Defaults to 'bounce-back'.

        genetic_mutation : Mutation, optional
            Pymoo's genetic algorithm's mutation operator after crossover. Defaults to NoMutation().
        
        repair : Repair, optional
            Pymoo's repair operator after mutation. Defaults to NoRepair().
        """

        # Default initialization of InfillCriterion
        super().__init__(eliminate_duplicates=None, **kwargs)

        # Parse the information from the string
        _, selection_variant, n_diff, crossover_variant, = variant.split("/")
        n_diffs = int(n_diff)

        # When "to" in variant there are more than 1 difference vectors
        if "-to-" in variant:
            n_diffs =  n_diffs + 1

        # Define parent selection operator
        self.selection = DES(selection_variant)
        
        # Define differential evolution mutation
        self.de_mutation = DEM(F=F, gamma=gamma, de_repair=de_repair, n_diffs=n_diffs)

        # Define crossover strategy (DE mutation is included)
        self.crossover = DEX(variant=crossover_variant, CR=CR, at_least_once=True)

        # Define posterior mutation strategy and repair
        self.genetic_mutation = genetic_mutation if genetic_mutation is not None else NoMutation()

    def _do(self, problem, pop, n_offsprings, **kwargs):

        # Select parents including donor vector
        parents = self.selection(
            problem, pop,
            n_offsprings, self.n_parents,
            to_pop=False, **kwargs,
        )

        # Mutant vectors from DE
        mutants = self.de_mutation(problem, pop, parents, **kwargs)
        
        # Crossover between mutant vector and target
        matings = self.merge_columnwise(pop, mutants)
        trials = self.crossover(problem, matings, **kwargs)

        # Perform posterior mutation and repair if passed
        off = self.genetic_mutation(problem, trials, **kwargs)

        return off
    
    @property
    def n_parents(self):
        return self.de_mutation.n_parents
    
    @staticmethod
    def merge_columnwise(*populations):
        matings = np.column_stack(populations).view(Population)
        return matings


# =========================================================================================================
# Implementation
# =========================================================================================================

class DifferentialEvolution(EvolutionaryAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=(0.5, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 survival=ImprovementReplacement(),
                 advance_after_initial_infill=True,
                 output=SingleObjectiveOutput(),
                 **kwargs):
        """
        Base class for Differential Evolution algorithms
        
        Parameters
        ----------
        pop_size : int, optional
            Population size. Defaults to 100.

        sampling : Sampling, optional
            Sampling strategy of pymoo. Defaults to LHS().

        variant : str, optional
            Differential evolution strategy. Must be a string in the format: "DE/selection/n/crossover", in which, n in an integer of number of difference vectors, and crossover is either 'bin' or 'exp'. Selection variants are:

                - 'ranked'
                - 'rand'
                - 'best'
                - 'current-to-best'
                - 'current-to-best'
                - 'current-to-rand'
                - 'rand-to-best'

            The selection strategy 'ranked' might be helpful to improve convergence speed without much harm to diversity. Defaults to 'DE/rand/1/bin'.

        CR : float, optional
            Crossover parameter. Defined in the range [0, 1]
            To reinforce mutation, use higher values. To control convergence speed, use lower values.

        F : iterable of float or float, optional
            Scale factor or mutation parameter. Defined in the range (0, 2]
            To reinforce exploration, use higher values; for exploitation, use lower values.

        gamma : float, optional
            Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.

        de_repair : str, optional
            Repair of DE mutant vectors. Is either callable or one of:

                - 'bounce-back'
                - 'midway'
                - 'rand-init'
                - 'to-bounds'

            If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors including violations, Xb contains reference vectors for repair in feasible space, xl is a 1d vector of lower bounds, and xu a 1d vector of upper bounds.
            Defaults to 'bounce-back'.

        genetic_mutation, optional
            Pymoo's genetic mutation operator after crossover. Defaults to NoMutation().
        
        survival : Survival, optional
            Replacement survival operator. Defaults to ImprovementReplacement().

        repair : Repair, optional
            Pymoo's repair operator after mutation. Defaults to NoRepair().
        """

        # Mating
        mating = DifferentialVariant(
            variant=variant, CR=CR, F=F, gamma=gamma,
            de_repair=de_repair, **kwargs,
        )

        # Number of offsprings at each generation
        n_offsprings = pop_size

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            mating=mating,
            n_offsprings=n_offsprings,
            eliminate_duplicates=None,
            survival=survival,
            output=output,
            advance_after_initial_infill=advance_after_initial_infill,
            **kwargs,
        )
        
        self.termination = DefaultSingleObjectiveTermination()

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(self.problem, infills, None, n_survive=self.pop_size)

    def _infill(self):
        infills = self.mating(self.problem, self.pop, self.n_offsprings, algorithm=self)
        return infills

    def _advance(self, infills=None, **kwargs):

        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must be provided."

        # One-to-one replacement survival
        self.pop = self.survival.do(self.problem, self.pop, infills)
    
    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


class MODE(DifferentialEvolution):
    
    def __init__(self,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.5,
                 F=None,
                 gamma=0.0001,
                 de_repair="bounce-back",
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            variant=variant,
            CR=CR,
            F=F,
            gamma=gamma,
            de_repair=de_repair,
            survival=survival,
            output=output,
            **kwargs,
        )
        
        self.termination = DefaultMultiObjectiveTermination()
    
    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(self.problem, infills, None, n_survive=self.pop_size)
