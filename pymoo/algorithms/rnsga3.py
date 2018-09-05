import numpy as np

from pymoo.algorithms.nsga3 import comp_by_cv_then_random, NSGA3
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.aspiration_point_survival import AspirationPointSurvival
from pymoo.util.display import disp_multi_objective
from pymoo.util.reference_directions import get_ref_dirs_from_n


class RNSGA3(NSGA3):

    def __init__(self,
                 ref_points,
                 pop_size,
                 mu=0.1,
                 prob_cross=1.0,
                 eta_cross=20,
                 prob_mut=None,
                 eta_mut=30,
                 **kwargs):
        self.ref_points = ref_points
        self.mu = mu

        set_if_none(kwargs, 'pop_size', pop_size)
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=prob_cross, eta_cross=eta_cross))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=prob_mut, eta_mut=eta_mut))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

    def _initialize(self):
        problem = self.D['problem']
        n_ref_points = self.ref_points.shape[0]

        # add the aspiration point lines
        aspiration_ref_dirs = []
        n_aspriation_ref_dirs_per_point = int(self.pop_size / n_ref_points)
        for i in range(n_ref_points):
            aspiration_ref_dirs.extend(get_ref_dirs_from_n(problem.n_obj, n_aspriation_ref_dirs_per_point))
        aspiration_ref_dirs = np.array(aspiration_ref_dirs)

        # create the survival strategy
        self.survival = AspirationPointSurvival(self.ref_points, aspiration_ref_dirs, mu=self.mu)

        pop = super()._initialize()

        # call the survival to initialize ideal point and so on - does not do a actual survival
        self.survival.do(pop, self.pop_size, D=self.D)

        return pop
