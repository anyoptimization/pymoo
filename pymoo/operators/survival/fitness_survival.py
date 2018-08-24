import numpy as np

from pymoo.model.survival import Survival, split_by_feasibility


class FitnessSurvival(Survival):

    def _do(self, pop, n_survive, out=None, **kwargs):

        if pop.F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective problems!")

        # split by feasibility
        feasible, infeasible = split_by_feasibility(pop)
        survivors = np.array([], dtype=np.int)

        # if there are feasible solutions add them first
        if len(feasible) > 0:
            survivors = feasible[np.argsort(pop.F[feasible, 0])]

        # if we can select only from feasible - truncate here
        if len(feasible) > n_survive:
            survivors = survivors[:n_survive]

        # otherwise fill up with least infeasible ones
        else:
            n_infeasible = (n_survive - len(feasible))
            survivors = np.concatenate([survivors, infeasible[:n_infeasible]])

        pop.filter(survivors)
