import numpy as np
from abc import abstractmethod
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.core.variable import get
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.expx import mut_exp


# =========================================================================================================
# Implementation
# =========================================================================================================

class DifferentialOperator(Crossover):

    def __init__(self, n_parents=None, **kwargs):
        """White label for differential evolution operators

        Parameters
        ----------
        n_parents : int | None, optional
            Number of parents necessary in its operations. Useful for compatibility with pymoo.
        """
        # __init__ operator
        super().__init__(n_parents=n_parents, n_offsprings=1, prob=1.0, **kwargs)

    @staticmethod
    def default_prepare(pop, parents):
        """Utility function that converts population and parents from pymoo Selection to pop and X

        Parameters
        ----------
        pop : Population
            pymoo population

        parents : Population | np.array (n_samples, n_parents) | None
            Parent population or indices

        Returns
        -------
        pop, X : Population (n_samples, n_parents), np.array (n_parents, n_samples, n_var)
            Population and corresponding decision variables
        """
        # Convert pop if parents is not None
        if parents is not None:
            pop = pop[parents]

        # Get all X values for mutation parents
        X = np.swapaxes(pop, 0, 1).get("X")
        return pop, X

    @abstractmethod
    def do(self, problem, pop, parents=None, **kwargs):
        pass

    @abstractmethod
    def _do(self, problem, X, **kwargs):
        pass


class DifferentialCrossover(DifferentialOperator):
    
    def __init__(self,
                 variant="bin",
                 CR=0.7,
                 at_least_once=True,
                 **kwargs):
        """Differential evolution crossover
        (DE mutation is considered a part of this operator)

        Parameters
        ----------
        variant : str | callable, optional
            Crossover variant. Must be either "bin", "exp", or callable. By default "bin".
            If callable, it has the form:
            ``cross_function(n_matings, n_var, CR, at_least_once=True)``
        
        CR : float, optional
            Crossover parameter. Defined in the range [0, 1]
            To reinforce mutation, use higher values. To control convergence speed, use lower values.

        at_least_once : bool, optional
            Either or not offsprings must inherit at least one attribute from mutant vectors, by default True
        """
        
        # __init__ operator
        super().__init__(n_parents=2, **kwargs)
        
        self.CR = CR
        self.variant = variant
        self.at_least_once = at_least_once
    
    def do(self, problem, pop, parents=None, **kwargs):
        
        # Convert pop if parents is not None
        pop, X = self.default_prepare(pop, parents)

        # Create child vectors
        U = self._do(problem, X, **kwargs)

        return Population.new("X", U)
    
    @abstractmethod
    def _do(self, problem, X, **kwargs):
        pass


class DEX(DifferentialCrossover):
    
    def __init__(self,
                 variant="bin",
                 CR=0.7,
                 at_least_once=True,
                 **kwargs):
        
        super().__init__(
            variant=variant, CR=CR,
            at_least_once=at_least_once,
            **kwargs,
        )
        
        if self.variant == "bin":
            self.cross_function = mut_binomial
        elif self.variant == "exp":
            self.cross_function = mut_exp
        elif hasattr(self.variant, "__call__"):
            self.cross_function = self.variant
        else:
            raise ValueError("Crossover variant must be either 'bin', 'exp', or callable")

    def _do(self, problem, X, **kwargs):
        
        # Decompose input vector
        V = X[1]
        X_ = X[0]
        U = np.array(X_, copy=True)
        
        # About X
        n_matings, n_var = X_.shape
        
        # Mask
        CR = get(self.CR, size=n_matings)
        M = self.cross_function(n_matings, n_var, CR, self.at_least_once)
        U[M] = V[M]
        
        return U
