import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.expx import mut_exp


# =========================================================================================================
# Implementation
# =========================================================================================================

class DEM:
    
    def __init__(self,
                 F=None,
                 gamma=1e-4,
                 de_repair="bounce-back",
                 **kwargs):

        # Default value for F
        if F is None:
            F = (0.0, 1.0)
        
        # Define which method will be used to generate F values
        if hasattr(F, "__iter__"):
            self.scale_factor = self._randomize_scale_factor
        else:
            self.scale_factor = self._scalar_scale_factor
        
        # Define which method will be used to generate F values
        if not hasattr(de_repair, "__call__"):
            try:
                de_repair = REPAIRS[de_repair]
            except:
                raise KeyError("Repair must be either callable or in " + str(list(REPAIRS.keys())))
        
        # Define which strategy of rotation will be used
        if gamma is None:
            self.get_diff = self._diff_simple
        else:
            self.get_diff = self._diff_jitter
        
        self.F = F
        self.gamma = gamma
        self.de_repair = de_repair
    
    def __call__(self, problem, pop, parents, **kwargs):
        return self.do(problem, pop, parents, **kwargs)
        
    def do(self, problem, pop, parents, **kwargs):

        #Get all X values for mutation parents
        Xr = pop.get("X")[parents.T].copy()
        assert len(Xr.shape) == 3, "Please provide a three-dimensional matrix n_parents x pop_size x n_vars."
        
        #Create mutation vectors
        V, diffs = self.de_mutation(Xr, return_differentials=True)

        # If the problem has boundaries to be considered
        if problem.has_bounds():
            
            # Do repair
            V = self.de_repair(V, Xr[0], *problem.bounds())
                
        return Population.new("X", V)
    
    def de_mutation(self, Xr, return_differentials=True):
        
        n_parents, n_matings, n_var = Xr.shape
        assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

        # Build the pairs for the differentials
        pairs = (np.arange(n_parents - 1) + 1).reshape(-1, 2)

        # The differentials from each pair subtraction
        diffs = self.get_diffs(Xr, pairs, n_matings, n_var)

        # Add the difference vectors to the base vector
        V = Xr[0] + diffs

        if return_differentials:
            return V, diffs
        else:
            return V
    
    def _randomize_scale_factor(self, n_matings):
         return (self.F[0] + np.random.random(n_matings) * (self.F[1] - self.F[0]))
     
    def _scalar_scale_factor(self, n_matings):
         return np.full(n_matings, self.F)
     
    def _diff_jitter(self, F, Xi, Xj, n_matings, n_var):
        F = F[:, None] * (1 + self.gamma * (np.random.random((n_matings, n_var)) - 0.5))
        return F * (Xi - Xj)
    
    def _diff_simple(self, F, Xi, Xj, n_matings, n_var):
        return F[:, None] * (Xi - Xj)
    
    def get_diffs(self, Xr, pairs, n_matings, n_var):
        
        # The differentials from each pair subtraction
        diffs = np.zeros((n_matings, n_var))
        
        # For each difference
        for i, j in pairs:
        
            # Obtain F randomized in range
            F = self.scale_factor(n_matings)
            
            # New difference vector
            diff = self.get_diff(F, Xr[i], Xr[j], n_matings, n_var)
            
            # Add the difference to the first vector
            diffs = diffs + diff
        
        return diffs
        
    
class DEX(Crossover):
    
    def __init__(self,
                 variant="bin",
                 CR=0.7,
                 F=None,
                 gamma=1e-4,
                 n_diffs=1,
                 at_least_once=True,
                 de_repair="bounce-back",
                 **kwargs):
        
        # Default value for F
        if F is None:
            F = (0.0, 1.0)
        
        # Create instace for mutation
        self.dem = DEM(F=F,
                       gamma=gamma,
                       de_repair=de_repair)
    
        self.CR = CR
        self.variant = variant
        self.at_least_once = at_least_once
        
        super().__init__(2 + 2 * n_diffs, 1,  prob=1.0, **kwargs)

    
    def do(self, problem, pop, parents, **kwargs):
        
        #Get target vectors
        X = pop.get("X")[parents[:, 0]]
        
        #About Xi
        n_matings, n_var = X.shape
        
        #Obtain mutants
        mutants = self.dem.do(problem, pop, parents[:, 1:], **kwargs)
        
        # Obtain V
        V = mutants.get("X")
        
        # Binomial crossover
        if self.variant == "bin":
            M = mut_binomial(n_matings, n_var, self.CR, at_least_once=self.at_least_once)
        # Exponential crossover
        elif self.variant == "exp":
            M = mut_exp(n_matings, n_var, self.CR, at_least_once=self.at_least_once)
        else:
            raise Exception(f"Unknown variant: {self.variant}")

        # Add mutated elements in corresponding main parent
        Xp[M] = V[M]

        off = Population.new("X", Xp)
        
        return off
    

def bounce_back(X, Xb, xl, xu):
    """Repair strategy

    Args:
        X (2d array like): Mutated vectors including violations.
        Xb (2d array like): Reference vectors for repair in feasible space.
        xl (1d array like): Lower-bounds.
        xu (1d array like): Upper-bounds.

    Returns:
        2d array like: Repaired vectors.
    """
    
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)

    i, j = np.where(X < XL)
    if len(i) > 0:
        X[i, j] = XL[i, j] + np.random.random(len(i)) * (Xb[i, j] - XL[i, j])

    i, j = np.where(X > XU)
    if len(i) > 0:
        X[i, j] = XU[i, j] - np.random.random(len(i)) * (XU[i, j] - Xb[i, j])

    return X

def midway(X, Xb, xl, xu):
    """Repair strategy

    Args:
        X (2d array like): Mutated vectors including violations.
        Xb (2d array like): Reference vectors for repair in feasible space.
        xl (1d array like): Lower-bounds.
        xu (1d array like): Upper-bounds.

    Returns:
        2d array like: Repaired vectors.
    """
    
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)

    i, j = np.where(X < XL)
    if len(i) > 0:
        X[i, j] = XL[i, j] + (Xb[i, j] - XL[i, j]) / 2

    i, j = np.where(X > XU)
    if len(i) > 0:
        X[i, j] = XU[i, j] - (XU[i, j] - Xb[i, j]) / 2

    return X

def to_bounds(X, Xb, xl, xu):
    """Repair strategy

    Args:
        X (2d array like): Mutated vectors including violations.
        Xb (2d array like): Reference vectors for repair in feasible space.
        xl (1d array like): Lower-bounds.
        xu (1d array like): Upper-bounds.

    Returns:
        2d array like: Repaired vectors.
    """
    
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)

    i, j = np.where(X < XL)
    if len(i) > 0:
        X[i, j] = XL[i, j]

    i, j = np.where(X > XU)
    if len(i) > 0:
        X[i, j] = XU[i, j]

    return X

def rand_init(X, Xb, xl, xu):
    """Repair strategy

    Args:
        X (2d array like): Mutated vectors including violations.
        Xb (2d array like): Reference vectors for repair in feasible space.
        xl (1d array like): Lower-bounds.
        xu (1d array like): Upper-bounds.

    Returns:
        2d array like: Repaired vectors.
    """
    
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)

    i, j = np.where(X < XL)
    if len(i) > 0:
        X[i, j] = XL[i, j] + np.random.random(len(i)) * (XU[i, j] - XL[i, j])

    i, j = np.where(X > XU)
    if len(i) > 0:
        X[i, j] = XU[i, j] - np.random.random(len(i)) * (XU[i, j] - XL[i, j])

    return X


def squared_bounce_back(X, Xb, xl, xu):
    """Repair strategy

    Args:
        X (2d array like): Mutated vectors including violations.
        Xb (2d array like): Reference vectors for repair in feasible space.
        xl (1d array like): Lower-bounds.
        xu (1d array like): Upper-bounds.

    Returns:
        2d array like: Repaired vectors.
    """
    
    XL = xl[None, :].repeat(len(X), axis=0)
    XU = xu[None, :].repeat(len(X), axis=0)

    i, j = np.where(X < XL)
    if len(i) > 0:
        X[i, j] = XL[i, j] + np.square(np.random.random(len(i))) * (Xb[i, j] - XL[i, j])

    i, j = np.where(X > XU)
    if len(i) > 0:
        X[i, j] = XU[i, j] - np.square(np.random.random(len(i))) * (XU[i, j] - Xb[i, j])

    return X

def normalize_fun(fun):
    
    fmin = fun.min(axis=0)
    fmax = fun.max(axis=0)
    den = fmax - fmin
    
    den[den <= 1e-16] = 1.0
    
    return (fun - fmin)/den

REPAIRS = {"bounce-back":bounce_back,
           "midway":midway,
           "rand-init":rand_init,
           "to-bounds":to_bounds}
