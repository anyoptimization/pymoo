from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowdingSurvival
from pymoo.rand import random
from pymoo.util.dominator import Dominator


class NSGAII(GeneticAlgorithm):
    def __init__(self, var_type, **kwargs):
        set_default_if_none(var_type, kwargs)
        set_if_none(kwargs, 'selection', TournamentSelection(f_comp=comp_by_rank_and_crowding))
        set_if_none(kwargs, 'survival', RankAndCrowdingSurvival())
        super().__init__(**kwargs)


def comp_by_rank_and_crowding(pop, indices, data):
    if len(indices) != 2:
        raise ValueError("Only implemented for binary tournament!")

    first = indices[0]
    second = indices[1]

    if data.rank[first] < data.rank[second]:
        return first
    elif data.rank[second] < data.rank[first]:
        return second
    else:
        if data.crowding[first] > data.crowding[second]:
            return first
        elif data.crowding[second] > data.crowding[first]:
            return second
        else:
            return indices[random.randint(0, 2)]


def comp_by_dom_and_crowding(pop, indices, data):
    if len(indices) != 2:
        raise ValueError("Only implemented for binary tournament!")

    first = indices[0]
    second = indices[1]

    rel = Dominator.get_relation(pop.F[first, :], pop.F[second, :])

    if rel == 1:
        return first
    elif rel == -1:
        return second
    else:
        if data.crowding[first] > data.crowding[second]:
            return first
        elif data.crowding[second] > data.crowding[first]:
            return second
        else:
            return indices[random.randint(0, 2)]
