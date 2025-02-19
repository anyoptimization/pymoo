import numpy as np

from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.survival import Survival


def is_better(_new, _old, eps=0.0):
    both_infeasible = not _old.feas and not _new.feas
    both_feasible = _old.feas and _new.feas

    if both_infeasible and _old.CV[0] - _new.CV[0] > eps:
        return True
    elif not _old.FEAS and _new.FEAS:
        return True
    elif both_feasible and _old.F[0] - _new.F[0] > eps:
        return True

    return False


class ReplacementSurvival(Survival):

    def do(self, problem, pop, off, return_indices=False, inplace=False, **kwargs):

        # this makes it usable as a traditional survival
        if isinstance(off, int):
            k = off
            off = pop[k:]
            pop = pop[:k]

        # if the offsprings are simply empty don't do anything
        if len(off) == 0:
            return pop

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
            return pop

    def _do(self, problem, pop, off, **kwargs):
        pass


class ImprovementReplacement(ReplacementSurvival):

    def _do(self, problem, pop, off, **kwargs):

        ret = np.full((len(pop), 1), False)

        pop_F, pop_CV, pop_feas = pop.get("F", "CV", "FEAS")
        off_F, off_CV, off_feas = off.get("F", "CV", "FEAS")

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
