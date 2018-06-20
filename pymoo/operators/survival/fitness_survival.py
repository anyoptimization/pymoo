import numpy as np

from pymoo.model.survival import Survival
from pymop.problem import Problem


class FitnessSurvival(Survival):
    """

    This survival method is just for single-objective algorithm.
    Simply sort by first constraint violation and then fitness value and truncate the worst individuals.


    """
    def _do(self, pop, size, data):

        if pop.F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective problems!")

        if pop.G is None or len(pop.G) == 0:
            CV = np.zeros(pop.F.shape[0])
        else:
            CV = Problem.calc_constraint_violation(pop.G)
            CV[CV < 0] = 0.0

        # sort by cv and fitness
        sorted_idx = sorted(range(pop.size()), key=lambda x: (CV[x], pop.F[x]))

        # now truncate the population
        sorted_idx = sorted_idx[:size]
        pop.filter(sorted_idx)

        return pop

