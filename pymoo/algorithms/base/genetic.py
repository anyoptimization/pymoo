from pymoo.algorithms.base.evolutionary import EvolutionaryAlgorithm
from pymoo.core.mating import Mating
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination


# =========================================================================================================
# Implementation
# =========================================================================================================

class GeneticAlgorithm(EvolutionaryAlgorithm):
    
    def __init__(self,
                 pop_size=None,
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=True,
                 repair=None,
                 **kwargs):
        """Base class for Genetic Algorithms. A Mating operator is instantiated using 
        selection, crossover, mutation, repair, and eliminate_duplicates arguments.

        Parameters
        ----------
        pop_size : int, optional
            Population size, by default None
        
        sampling : Sampling, optional
            pymoo Sampling instance, by default None
        
        selection : Selection, optional
            pymoo parent selection operator, by default None
        
        crossover : Crossover, optional
            pymoo crossover operator, by default None
        
        genetic_mutation, optional
            pymoo mutation operator, by default None
        
        survival : Survival, optional
            pymoo survival operator, by default None
        
        n_offsprings : int, optional
            Number of offspring individuals created at each generation, by default None
        
        eliminate_duplicates : DuplicateElimination | bool | None, optional
            Eliminate duplicates in mating, by default True
        
        repair : Repair, optional
            pymoo repair operator which should be passed to Mating and self. By default None
        
        advance_after_initial_infill : bool, optional
            Either or not apply survival after initialization, by default False
        """
        
        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                eliminate_duplicates = DefaultDuplicateElimination()
            else:
                eliminate_duplicates = NoDuplicateElimination()
        
        # Mating operator of genetic algorithms
        mating = Mating(selection, crossover, mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates)
        
        # Instantiate generic evolutionary algorithm
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            survival=survival,
            mating=mating,
            n_offsprings=n_offsprings,
            eliminate_duplicates=eliminate_duplicates,
            repair=repair, 
            **kwargs,
        )
