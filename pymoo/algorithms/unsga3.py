import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.docs import parse_doc_string
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.rand import random


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
            S[i] = random.choice([a, b])

    return S[:, None].astype(np.int)


# =========================================================================================================
# Interface
# =========================================================================================================

def unsga3(**kwargs):
    """
    This is an implementation of the Unified NSGA3 algorithm :cite:`unsga3`. The same options as for
    :class:`pymoo.algorithms.nsga3.nsga3` are available.

    Returns
    -------
    unsga3 : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an UNSGA3 algorithm object.


    """

    return NSGA3(selection=TournamentSelection(func_comp=comp_by_rank_and_ref_line_dist), **kwargs)


parse_doc_string(unsga3)
