import numpy as np

from pymoo.model.individual import Individual
from pymoo.model.population import Population


def is_better(_new, _old, eps=1e-8):
    both_infeasible = not _old.feasible[0] and not _new.feasible[0]
    both_feasible = _old.feasible[0] and _new.feasible[0]

    if both_infeasible and _old.CV[0] - _new.CV[0] > eps:
        return True
    elif not _old.feasible and _new.feasible:
        return True
    elif both_feasible and _old.F[0] - _new.F[0] > eps:
        return True

    return False


class ReplacementStrategy:

    def do(self, problem, pop, off, return_indices=False, inplace=False, **kwargs):
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


class ImprovementReplacement(ReplacementStrategy):

    def _do(self, problem, pop, off, **kwargs):
        ret = np.full((len(pop), 1), False)

        pop_F, pop_CV, pop_feasible = pop.get("F", "CV", "feasible")
        off_F, off_CV, off_feasible = off.get("F", "CV", "feasible")

        if problem.n_constr > 0:

            # 1) Both infeasible and constraints have been improved
            ret[(~pop_feasible & ~off_feasible) & (off_CV < pop_CV)] = True

            # 2) A solution became feasible
            ret[~pop_feasible & off_feasible] = True

            # 3) Both feasible but objective space value has improved
            ret[(pop_feasible & off_feasible) & (off_F < pop_F)] = True

        else:
            ret[off_F < pop_F] = True

        return ret[:, 0]
