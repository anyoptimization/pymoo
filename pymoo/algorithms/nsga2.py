import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover_c import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowdingSurvival
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import Dominator


class NSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 prob_cross=0.9,
                 eta_cross=15,
                 prob_mut=None,
                 eta_mut=20,
                 **kwargs):

        set_if_none(kwargs, 'pop_size', pop_size)
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'selection', TournamentSelection(f_comp=comp_by_dom_and_crowding))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=prob_cross, eta_cross=eta_cross))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=prob_mut, eta_mut=eta_mut))
        set_if_none(kwargs, 'survival', RankAndCrowdingSurvival())
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)
        self.func_display_attrs = disp_multi_objective

    def _initialize(self):
        pop = super()._initialize()
        # after initializing the rank and crowding needs to be calculated for the tournament selection
        self.survival.do(pop, self.pop_size, out=self.D, **self.D)
        return pop


def comp_by_rank_and_crowding(pop, P, rank, crowding, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.zeros((P.shape[0]), dtype=np.int)
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

            # first by rank
            if rank[a] < rank[b]:
                S[i] = a
            elif rank[b] < rank[a]:
                S[i] = b

            # then by crowding
            else:
                if crowding[a] > crowding[b]:
                    S[i] = a
                elif crowding[b] > crowding[a]:
                    S[i] = b
                else:
                    if random.random() < 0.5:
                        S[i] = a
                    else:
                        S[i] = b

    return S[:, None]


def comp_by_dom_and_crowding(pop, P, crowding, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.zeros((P.shape[0]), dtype=np.int)
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

            rel = Dominator.get_relation(pop.F[a, :], pop.F[b, :])

            # first by domination
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b

            # then by crowding
            else:
                if crowding[a] > crowding[b]:
                    S[i] = a
                elif crowding[b] > crowding[a]:
                    S[i] = b
                else:
                    if random.random() < 0.5:
                        S[i] = a
                    else:
                        S[i] = b

    return S[:, None]
