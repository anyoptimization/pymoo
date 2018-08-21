import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.rand import random


class UNSGA3(NSGA3):

    def _initialize(self):
        pop = super()._initialize()

        # to set the rank and niches
        self.survival.do(pop, self.pop_size, out=self.D, **self.D)

        # add selection pressure to improve convergence
        self.selection = TournamentSelection(f_comp=comp_by_rank_and_ref_line_dist)

        return pop


def comp_by_rank_and_ref_line_dist(pop, P, niche, rank, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.zeros((P.shape[0]), dtype=np.int)

    # boolean array for feasibility
    feasible = pop.CV[:, 0] <= 0

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        if feasible[a] and not feasible[b]:
            S[i] = a
        elif not feasible[a] and feasible[b]:
            S[i] = b
        elif not feasible[b] and not feasible[a]:

            if pop.CV[a, 0] < pop.CV[b, 0]:
                S[i] = a
            elif pop.CV[b, 0] < pop.CV[a, 0]:
                S[i] = b
            else:
                if random.random() < 0.5:
                    S[i] = a
                else:
                    S[i] = b

        # both are feasible
        else:

            if niche[a] == niche[b]:

                if rank[a] < rank[b]:
                    S[i] = a
                elif rank[b] < rank[a]:
                    S[i] = b
                else:
                    if random.random() < 0.5:
                        S[i] = a
                    else:
                        S[i] = b

            else:

                if random.random() < 0.5:
                    S[i] = a
                else:
                    S[i] = b

    return S[:, None]
