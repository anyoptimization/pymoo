import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.reference_line_survival import ReferenceLineSurvival
from pymoo.rand import random
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import compare
from pymoo.util.reference_directions import get_ref_dirs_from_section, get_multi_layer_ref_dirs


class NSGA3(GeneticAlgorithm):

    def __init__(self,
                 n_sections=None,
                 ref_dirs=None,
                 prob_cross=1.0,
                 eta_cross=20,
                 prob_mut=None,
                 eta_mut=30,
                 **kwargs):

        self.ref_dirs = ref_dirs
        self.n_sections = n_sections

        # at least one of both must be provided
        if self.ref_dirs is None and self.n_sections is None:
            raise Exception("Either provide the reference lines directly or the number of sections for the uniform "
                            "reference line sampling!")

        if self.ref_dirs is not None:
            kwargs['pop_size'] = ref_dirs.shape[0]
        else:
            kwargs['pop_size'] = -1

        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=prob_cross, eta_cross=eta_cross))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=prob_mut, eta_mut=eta_mut))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

    def _initialize(self):

        # if survival not define differently
        if self.survival is None:

            # if ref dirs are not initialized do it based on the population size
            if self.ref_dirs is None:

                if isinstance(self.n_sections, int):
                    self.ref_dirs = get_ref_dirs_from_section(self.problem.n_obj, self.n_sections)
                elif isinstance(self.n_sections, list):
                    self.ref_dirs = get_multi_layer_ref_dirs(self.problem.n_obj, self.n_sections)
                else:
                    raise Exception("n_section must be either an integer or list [(p1, scaling1), (p2, scaling2), ..]")

                self.pop_size = self.ref_dirs.shape[0]
                if self.n_offsprings == -1:
                    self.n_offsprings = self.pop_size

            # set the survival method itself
            self.survival = ReferenceLineSurvival(self.ref_dirs)

        pop = super()._initialize()

        # call the survival to initialize ideal point and so on - does not do a actual survival
        self.survival.do(pop, self.pop_size, out=self.D, **self.D)

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
