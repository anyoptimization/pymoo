import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.rand import random
from pymoo.util.dominator import compare


class UNSGA3(NSGA3):

    def _initialize(self):
        pop = super()._initialize()

        # add selection pressure to improve convergence
        self.selection = TournamentSelection(f_comp=comp_by_rank_and_ref_line_dist)

        return pop


def comp_by_rank_and_ref_line_dist(pop, P, niche, rank, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop.CV[a, 0] > 0.0 or pop.CV[b, 0] > 0.0:
            S[i] = compare(a, pop.CV[a, 0], b, pop.CV[b, 0], method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:

            # if in the same niche select by rank
            if niche[a] == niche[b]:
                S[i] = compare(a, rank[a], b, rank[b], method='smaller_is_better', return_random_if_equal=True)

            # otherwise just select random
            else:
                S[i] = random.choice([a, b])

    return S[:, None].astype(np.int)

