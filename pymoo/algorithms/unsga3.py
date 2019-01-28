import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.rand import random


class UNSGA3(NSGA3):

    def __init__(self,
                 ref_dirs,
                 **kwargs):

        if 'pop_size' not in kwargs:
            kwargs['pop_size'] = ref_dirs.shape[0]
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_rank_and_ref_line_dist))
        super().__init__(ref_dirs, **kwargs)


def comp_by_rank_and_ref_line_dist(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:

            # if in the same niche select by rank
            if pop[a].get("niche") == pop[b].get("niche"):
                S[i] = compare(a, pop[a].get("niche"), b, pop[b].get("niche"), method='smaller_is_better',
                               return_random_if_equal=True)

            # otherwise just select random
            else:
                S[i] = random.choice([a, b])

    return S[:, None].astype(np.int)
