import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.selection.tournament import compare, TournamentSelection


# =========================================================================================================
# Implementation
# =========================================================================================================


def comp_by_rank_and_ref_line_dist(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            # if in the same niche select by rank
            if pop[a].get("niche") == pop[b].get("niche"):

                if pop[a].get("rank") != pop[b].get("rank"):
                    S[i] = compare(a, pop[a].get("rank"), b, pop[b].get("rank"), method='smaller_is_better')

                else:
                    S[i] = compare(a, pop[a].get("dist_to_niche"), b, pop[b].get("dist_to_niche"),
                                   method='smaller_is_better')

        if np.isnan(S[i]):
            S[i] = np.random.choice([a, b])

    return S[:, None].astype(int)


class UNSGA3(NSGA3):

    def __init__(self,
                 ref_dirs,
                 selection=TournamentSelection(func_comp=comp_by_rank_and_ref_line_dist),
                 **kwargs):
        super().__init__(ref_dirs, selection=selection, **kwargs)
