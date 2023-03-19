# External
import numpy as np
from abc import abstractmethod
from pymoo.core.population import Population
from pymoo.operators.crossover.dex import DifferentialOperator


# =========================================================================================================
# Implementation
# =========================================================================================================

class DifferentialMutation(DifferentialOperator):

    def __init__(self,
                 F=None,
                 gamma=1e-4,
                 de_repair="bounce-back",
                 n_diffs=1,
                 **kwargs):
        """Differential Evolution mutation
        (Implemented similar to pymoo crossover)

        Parameters
        ----------
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

            If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors including violations,
            Xb contains reference vectors for repair in feasible space,
            xl is a 1d vector of lower bounds, and xu a 1d vector of upper bounds.
            Defaults to 'bounce-back'.
        
        n_diffs : int, optional
            Number of difference vectors in operation, by default 1

        Raises
        ------
        KeyError
            Bad de_repair input
        """
        # __init__ operator
        n_parents = 1 + 2 * n_diffs
        super().__init__(n_parents=n_parents, **kwargs)

        # Other attributes
        self.F = F
        self.gamma = gamma
        self.de_repair = de_repair

    def do(self, problem, pop, parents=None, **kwargs):
        """Perform Differential Evolution mutation

        Parameters
        ----------
        problem : Problem
            Optimization problem
        
        pop : Population
            Original parent population at a given generation
        
        parents : numpy.array (n_offsprings, n_select) of dtype int | None, optional
            Parent (indexes) to crossover. 
            If None, the argument pop should already be sampled as parents to crossover. 
            By default None

        Returns
        -------
        Population
            Offspring mutants V
        """
        # Convert pop if parents is not None
        pop, X = self.default_prepare(pop, parents)

        # Create mutation vectors
        V = self._do(problem, X, **kwargs)

        # If the problem has boundaries to be considered
        if problem.has_bounds():

            # Do de_repair
            V = self.de_repair(V, X[0], *problem.bounds())

        return Population.new("X", V)
    
    @abstractmethod
    def _do(self, problem, X, **kwargs):
        pass


class DEM(DifferentialMutation):

    def __init__(self,
                 F=None,
                 gamma=1e-4,
                 de_repair="bounce-back",
                 n_diffs=1,
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
            except Exception:
                raise KeyError("Repair must be either callable or in " + str(list(REPAIRS.keys())))

        # Define which strategy of rotation will be used
        if gamma is None:
            self.get_diff = self._diff_simple
        else:
            self.get_diff = self._diff_jitter

        super().__init__(
            F=F, gamma=gamma, de_repair=de_repair,
            n_diffs=n_diffs, **kwargs,
        )

    def _do(self, problem, X, **kwargs):
        return self.de_mutation(X, return_differentials=False)

    def de_mutation(self, X, return_differentials=True):
        """Perform DE mutation operator

        Parameters
        ----------
        X : numpy.array (n_parents, n_matings, n_var)
            Three dimensional array of parents selected to take part in DE mutation
        
        return_differentials : bool, optional
            Either or not to return a tuple of ``(V, diffs)``, by default True

        Returns
        -------
        np.array (n_matings, n_var) | tuple
            Mutant vectors created or pair of both mutant vectors and perturbation vectors
        """

        n_parents, n_matings, n_var = X.shape
        assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

        # Build the pairs for the differentials
        pairs = (np.arange(n_parents - 1) + 1).reshape(-1, 2)

        # The differentials from each pair subtraction
        diffs = self.get_diffs(X, pairs, n_matings, n_var)

        # Add the difference vectors to the base vector
        V = X[0] + diffs

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

    def get_diffs(self, X, pairs, n_matings, n_var):

        # The differentials from each pair subtraction
        diffs = np.zeros((n_matings, n_var))

        # For each difference
        for i, j in pairs:

            # Obtain F randomized in range
            F = self.scale_factor(n_matings)

            # New difference vector
            diff = self.get_diff(F, X[i], X[j], n_matings, n_var)

            # Add the difference to the first vector
            diffs = diffs + diff

        return diffs


# =========================================================================================================
# Repairs
# =========================================================================================================


def bounce_back(X, Xb, xl, xu):
    """Repair strategy that radomly re-initializes violated elements between reference points and bounds

    Parameters
    ----------
    X : numpy.array (n_samples, n_var)
        Original population in decision space (including violations)
    
    Xb : numpy.array (n_samples, n_var)
        Feasible reference points in decision space
    
    xl : numpy.array (n_var,)
        Lower bounds for decision variables
    
    xu : numpy.array (n_var,)
        Upper bounds for decision variables

    Returns
    -------
    numpy.array (n_samples, n_var)
        Repaired population
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
    """Repair strategy that sets violated elements to midpoint between reference points and bounds

    Parameters
    ----------
    X : numpy.array (n_samples, n_var)
        Original population in decision space (including violations)
    
    Xb : numpy.array (n_samples, n_var)
        Feasible reference points in decision space
    
    xl : numpy.array (n_var,)
        Lower bounds for decision variables
    
    xu : numpy.array (n_var,)
        Upper bounds for decision variables

    Returns
    -------
    numpy.array (n_samples, n_var)
        Repaired population
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
    """Repair strategy that forces violated elements to problem bounds

    Parameters
    ----------
    X : numpy.array (n_samples, n_var)
        Original population in decision space (including violations)
    
    Xb : numpy.array (n_samples, n_var)
        Feasible reference points in decision space
    
    xl : numpy.array (n_var,)
        Lower bounds for decision variables
    
    xu : numpy.array (n_var,)
        Upper bounds for decision variables

    Returns
    -------
    numpy.array (n_samples, n_var)
        Repaired population
    """

    xl = np.array(xl)
    xu = np.array(xu)

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
    """Repair strategy that radomly re-initializes violated elements to problem domain

    Parameters
    ----------
    X : numpy.array (n_samples, n_var)
        Original population in decision space (including violations)
    
    Xb : numpy.array (n_samples, n_var)
        Feasible reference points in decision space
    
    xl : numpy.array (n_var,)
        Lower bounds for decision variables
    
    xu : numpy.array (n_var,)
        Upper bounds for decision variables

    Returns
    -------
    numpy.array (n_samples, n_var)
        Repaired population
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
    """Repair strategy that radomly re-initializes violated elements between reference points and bounds
    closer to bounds

    Parameters
    ----------
    X : numpy.array (n_samples, n_var)
        Original population in decision space (including violations)
    
    Xb : numpy.array (n_samples, n_var)
        Feasible reference points in decision space
    
    xl : numpy.array (n_var,)
        Lower bounds for decision variables
    
    xu : numpy.array (n_var,)
        Upper bounds for decision variables

    Returns
    -------
    numpy.array (n_samples, n_var)
        Repaired population
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


REPAIRS = {"bounce-back": bounce_back,
           "midway": midway,
           "rand-init": rand_init,
           "to-bounds": to_bounds}
