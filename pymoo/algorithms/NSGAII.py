import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowdingSurvival
from pymoo.rand import random
from pymoo.util.dominator import Dominator


class NSGAII(GeneticAlgorithm):
    def __init__(self, var_type, **kwargs):
        set_if_none(kwargs, 'selection', TournamentSelection(f_comp=comp_by_rank_and_crowding))
        set_if_none(kwargs, 'survival', RankAndCrowdingSurvival())
        set_default_if_none(var_type, kwargs)
        super().__init__(**kwargs)


def comp_by_rank_and_crowding(pop, P, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    rank = kwargs['data']['rank']
    crowding = kwargs['data']['crowding']

    # the winner of the tournament selection
    S = np.zeros((P.shape[0], 1), dtype=np.int)

    for i, p in enumerate(P):

        # first by rank
        if rank[P[i, 0]] < rank[P[i, 1]]:
            S[i, 0] = P[i, 0]
        elif rank[P[i, 1]] < rank[P[i, 0]]:
            S[i, 0] = P[i, 1]

        # then by crowding
        else:
            if crowding[P[i, 0]] > crowding[P[i, 1]]:
                S[i, 0] = P[i, 0]
            elif crowding[P[i, 1]] > crowding[P[i, 0]]:
                S[i, 0] = P[i, 1]
            else:
                S[i, 0] = P[i, random.randint(0, 2)]
    return S


def comp_by_dom_and_crowding(pop, P, **kwargs):

    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    crowding = kwargs['data']['crowding']
    S = np.zeros((P.shape[0], 1), dtype=np.int)

    for i, p in enumerate(P):

        rel = Dominator.get_relation(pop.F[P[i, 0], :], pop.F[P[i, 1], :])

        # first by domination
        if rel == 1:
            S[i, 0] = P[i, 0]
        elif rel == -1:
            S[i, 0] = P[i, 1]

        # then by crowding
        else:
            if crowding[P[i, 0]] > crowding[P[i, 1]]:
                S[i, 0] = P[i, 0]
            elif crowding[P[i, 1]] > crowding[P[i, 0]]:
                S[i, 0] = P[i, 1]
            else:
                S[i, 0] = P[i, random.randint(0, 2)]
    return S