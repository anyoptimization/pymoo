import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover_c import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.reference_line_survival import ReferenceLineSurvival
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import compare
from pymoo.util.reference_directions import get_uniform_weights


class NSGA3(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 ref_dirs=None,
                 prob_cross=1.0,
                 eta_cross=20,
                 prob_mut=None,
                 eta_mut=30,
                 **kwargs):

        self.ref_dirs = ref_dirs

        set_if_none(kwargs, 'pop_size', pop_size)
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'selection', TournamentSelection(f_comp=comp_by_cv_then_random))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=prob_cross, eta_cross=eta_cross))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=prob_mut, eta_mut=eta_mut))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

    def _initialize(self):
        pop = super()._initialize()

        # if survival not define differently
        if self.survival is None:

            # if ref dirs are not initialized do it based on the population size
            if self.ref_dirs is None:
                self.ref_dirs = get_uniform_weights(self.pop_size, self.problem.n_obj)

            # set the survival method itself
            self.survival = ReferenceLineSurvival(self.ref_dirs)

        # call the survival to initialize ideal point and so on - does not do a actual survival
        self.survival.do(pop, pop.size())

        return pop


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
