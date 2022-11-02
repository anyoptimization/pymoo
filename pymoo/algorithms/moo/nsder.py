import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.misc import has_feasible
from pymoo.algorithms.moo.nsde import NSDE

# =========================================================================================================
# Implementation
# =========================================================================================================

class NSDER(NSDE):
    
    def __init__(self,
                 ref_dirs,
                 pop_size=100,
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=None,
                 gamma=1e-4,
                 **kwargs):
        """
        NSDE-R is an extension of NSDE to many-objective problems (Reddy & Dulikravich, 2019) using NSGA-III survival.
        
        S. R. Reddy and G. S. Dulikravich, "Many-objective differential evolution optimization based on reference points: NSDE-R," Struct. Multidisc. Optim., vol. 60, pp. 1455-1473, 2019.

        Parameters
        ----------
        ref_dirs : array like
            The reference directions that should be used during the optimization.
        
        pop_size : int, optional
            Population size. Defaults to 100.
            
        sampling : Sampling, optional
            Sampling strategy of pymoo. Defaults to LHS().
            
        variant : str, optional
            Differential evolution strategy. Must be a string in the format: "DE/selection/n/crossover", in which, n in an integer of number of difference vectors, and crossover is either 'bin' or 'exp'. Selection variants are:
            
                - "ranked'
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
            
        survival : Survival, optional
            Pymoo's survival strategy.
            Defaults to ReferenceDirectionSurvival().
        """
        
        self.ref_dirs = ref_dirs

        if self.ref_dirs is not None:

            if pop_size is None:
                pop_size = len(self.ref_dirs)

            if pop_size < len(self.ref_dirs):
                print(
                    f"WARNING: pop_size={pop_size} is less than the number of reference directions ref_dirs={len(self.ref_dirs)}.\n"
                    "This might cause unwanted behavior of the algorithm. \n"
                    "Please make sure pop_size is equal or larger than the number of reference directions. ")

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvival(ref_dirs)
            
        super().__init__(pop_size=pop_size,
                         variant=variant,
                         CR=CR,
                         F=F,
                         gamma=gamma,
                         survival=survival,
                         **kwargs)

    def _setup(self, problem, **kwargs):

        if self.ref_dirs is not None:
            if self.ref_dirs.shape[1] != problem.n_obj:
                raise Exception(
                    "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                    (self.ref_dirs.shape[1], problem.n_obj))
    
    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.survival.opt
