import numpy as np

from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.core.fitness import FitnessSurvival


def is_better(_new, _old, eps=0.0):
    both_infeasible = not _old.feasible[0] and not _new.feasible[0]
    both_feasible = _old.feasible[0] and _new.feasible[0]

    if both_infeasible and _old.CV[0] - _new.CV[0] > eps:
        return True
    elif not _old.feasible and _new.feasible:
        return True
    elif both_feasible and _old.F[0] - _new.F[0] > eps:
        return True

    return False


class ReplacementSurvival(Survival):
    
    def __init__(self, fitness=None):
        """Base class for Replacement survival

        Parameters
        ----------
        fitness : Survival, optional
            Survival used in ranking and fitness assignement, by default None
        """
        super().__init__(filter_infeasible=False)
        if fitness is None:
            fitness = FitnessSurvival()
        self.fitness = fitness

    def do(self, problem, pop, off, return_indices=False, inplace=False, **kwargs):
        """Given a problem a parent population and offspring canditates return next generation.
        By default, only fitness assigment is performed on parents. To define which ones should be replaced,
        define method ``_do`` in child classes that returns a boolean if parent should be replaced.

        Parameters
        ----------
        problem : Problem
            Pymoo problem
        
        pop : Population
            Parent population
        
        off : Population | None
            Offspring candidates. If None, only fitness assigment is performed on parents.
        
        return_indices : bool, optional
            Either or not to just return boolean of positions to be replaced, by default False
        
        inplace : bool, optional
            Change population inplace, by default False

        Returns
        -------
        Population
            Population that proceeds into the next generation
        """
        
        # If no offspring is available just do fitness assignment
        if (off is None) or len(off) == 0:
            return self.fitness.do(problem, pop, n_survive=None)

        # this makes it usable as a traditional survival
        elif isinstance(off, int) or isinstance(off, np.integer):
            k = off
            off = pop[k:]
            pop = pop[:k]

        assert len(pop) == len(off), "For the replacement pop and off must have the same number of individuals."

        pop = Population.create(pop) if isinstance(pop, Individual) else pop
        off = Population.create(off) if isinstance(off, Individual) else off

        I = self._do(problem, pop, off, **kwargs)

        if return_indices:
            return I
        else:
            if not inplace:
                pop = pop.copy()
            pop[I] = off[I]
            return self.fitness.do(problem, pop, n_survive=None)

    def _do(self, problem, pop, off, **kwargs):
        return np.zeros(len(pop), dtype=np.bool_)


class ImprovementReplacement(ReplacementSurvival):
    
    def do(self, problem, pop, off, return_indices=False, inplace=False, **kwargs):
        """Given a problem a parent population and offspring canditates return next generation,
        it selects those that proceed into the next generation via one-to-one comparison.
        Feasible solutions are always preferred to infeasible and infeasible solutions are compared by
        overall constraint violation.

        Parameters
        ----------
        problem : Problem
            Pymoo problem
        
        pop : Population
            Parent population
        
        off : Population | None
            Offspring candidates. If None, only fitness assigment is performed on parents.
        
        return_indices : bool, optional
            Either or not to just return boolean of positions to be replaced, by default False
        
        inplace : bool, optional
            Change population inplace, by default False

        Returns
        -------
        Population
            Population that proceeds into the next generation
        """
        return super().do(problem, pop, off, return_indices=False, inplace=False, **kwargs)

    def _do(self, problem, pop, off, **kwargs):

        ret = np.full((len(pop), 1), False)

        pop_F, pop_CV, pop_feas = pop.get("F", "CV", "feasible")
        off_F, off_CV, off_feas = off.get("F", "CV", "feasible")

        if problem.has_constraints() > 0:

            # 1) Both infeasible and constraints have been improved
            ret[(~pop_feas & ~off_feas) & (off_CV < pop_CV)] = True

            # 2) A solution became feasible
            ret[~pop_feas & off_feas] = True

            # 3) Both feasible but objective space value has improved
            ret[(pop_feas & off_feas) & (off_F < pop_F)] = True

        else:
            ret[off_F < pop_F] = True

        # never allow duplicates to become part of the population when replacement is used
        _, _, is_duplicate = DefaultDuplicateElimination(epsilon=0.0).do(off, pop, return_indices=True)
        ret[is_duplicate] = False

        return ret[:, 0]


def parameter_less(f, cv):
    v = np.copy(f)
    infeas = cv > 0
    v[infeas] = f.max() + cv[infeas]
    return v


def hierarchical_sort(f, cv=None):
    if cv is not None:
        f = parameter_less(f, cv)
    return np.argsort(f)
