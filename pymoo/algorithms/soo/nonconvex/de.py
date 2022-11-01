import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.replacement import ImprovementReplacement
from pymoo.operators.mutation.nom import NoMutation
from pymoo.core.repair import NoRepair
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.operators.selection.des import DES
from pymoo.operators.crossover.dex import DEX


# =========================================================================================================
# Implementation
# =========================================================================================================

class InfillDE:
    
    def __init__(self,
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=(0.5, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 mutation=None,
                 repair=None):
        
        # Parse the information from the string
        _, selection_variant, n_diff, crossover_variant, = variant.split("/")
        n_diffs = int(n_diff)
        
        # When "to" in variant there are more than 1 difference vectors
        if "-to-" in variant:
            n_diffs += 1
        
        # Define parent selection operator
        self.selection = DES(selection_variant)
        
        #Default value for F
        if F is None:
            F = (0.0, 1.0)
        
        # Define crossover strategy (DE mutation is included)
        self.crossover = DEX(variant=crossover_variant,
                             CR=CR,
                             F=F,
                             gamma=gamma,
                             n_diffs=n_diffs,
                             at_least_once=True,
                             de_repair=de_repair)
        
        # Define posterior mutation strategy and repair
        self.mutation = mutation if mutation is not None else NoMutation()
        self.repair = repair if repair is not None else NoRepair()

    def do(self, problem, pop, n_offsprings, **kwargs):
        
        # Select parents including donor vector
        parents = self.selection(problem, pop, n_offsprings, self.crossover.n_parents, to_pop=True, **kwargs)
        
        # Perform mutation included in DEX and crossover
        off = self.crossover(problem, parents, **kwargs)
        
        # Perform posterior mutation and repair if passed
        off = self.mutation(problem, off)
        off = self.repair(problem, off)
        
        return off
        

class DE(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=(0.5, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 mutation=None,
                 repair=None,
                 output=SingleObjectiveOutput(),
                 **kwargs):
        """
        Single-objective Differential Evolution proposed by Storn and Price (1997).

        Storn, R. & Price, K., 1997. Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. J. Glob. Optim., 11(4), pp. 341-359.

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
        
        mutation : Mutation, optional
            Pymoo's mutation operator after crossover. Defaults to NoMutation().
        
        repair : Repair, optional
            Pymoo's repair operator after mutation. Defaults to NoRepair().
        """
        
        mating = InfillDE(variant=variant,
                          CR=CR,
                          F=F,
                          gamma=gamma,
                          de_repair=de_repair,
                          mutation=mutation,
                          repair=repair)
        
        # Number of offsprings at each generation
        n_offsprings = pop_size

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         mating=mating,
                         n_offsprings=n_offsprings,
                         eliminate_duplicates=False,
                         output=output,
                         **kwargs)

        self.termination = DefaultSingleObjectiveTermination()

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = FitnessSurvival().do(self.problem, infills, n_survive=self.pop_size)

    def _infill(self):
        
        infills = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        return infills

    def _advance(self, infills=None, **kwargs):
        
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must be provided."

        #One-to-one replacement survival
        self.pop = ImprovementReplacement().do(self.problem, self.pop, infills)

        #Sort the population by fitness to make the selection simpler for mating (not an actual survival, just sorting)
        self.pop = FitnessSurvival().do(self.problem, self.pop)
        
        #Set ranks
        self.pop.set("rank", np.arange(self.pop_size))
