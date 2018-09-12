import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.c_reference_line_survival import ReferenceLineSurvival
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import compare


class NSGA3(GeneticAlgorithm):

    def __init__(self, ref_dirs, **kwargs):

        self.ref_dirs = ref_dirs
        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=None, eta_mut=30))
        set_if_none(kwargs, 'survival', ReferenceLineSurvival(ref_dirs))
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

    def _solve(self, problem, termination):
        if self.ref_dirs.shape[1] != problem.n_obj:
            raise Exception(
                "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                (self.ref_dirs.shape[1], problem.n_obj))

        return super()._solve(problem, termination)


def comp_by_cv_then_random(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop.CV[a, 0] > 0.0 or pop.CV[b, 0] > 0.0:
            S[i] = compare(a, pop.CV[a, 0], b, pop.CV[b, 0], method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = random.choice([a, b])

    return S[:, None].astype(np.int)
