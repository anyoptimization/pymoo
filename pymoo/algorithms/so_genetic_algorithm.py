from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.operators.survival.fitness_survival import FitnessSurvival

from pymoo.util.display import disp_single_objective


class SingleObjectiveGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, pop_size=100,
                 prob_cross=0.9,
                 eta_cross=3,
                 prob_mut=None,
                 eta_mut=5,
                 **kwargs):

        set_if_none(kwargs, 'pop_size', pop_size)
        set_if_none(kwargs, 'sampling', RealRandomSampling())
        set_if_none(kwargs, 'selection', RandomSelection())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=prob_cross, eta_cross=eta_cross))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=prob_mut, eta_mut=eta_mut))
        set_if_none(kwargs, 'survival', FitnessSurvival())
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)
        self.func_display_attrs = disp_single_objective