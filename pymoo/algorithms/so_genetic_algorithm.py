import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.fitness_survival import FitnessSurvival
from pymoo.util.display import disp_single_objective
from pymoo.util.dominator import compare


class SingleObjectiveGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, **kwargs):

        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_and_fitness))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=0.9, eta_cross=3))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=None, eta_mut=5))
        set_if_none(kwargs, 'survival', FitnessSurvival())
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)
        self.func_display_attrs = disp_single_objective


def comp_by_cv_and_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop.CV[a, 0] > 0.0 or pop.CV[b, 0] > 0.0:
            S[i] = compare(a, pop.CV[a, 0], b, pop.CV[b, 0], method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = compare(a, pop.F[a, 0], b, pop.F[b, 0], method='smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int)
