from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.fitness_survival import FitnessSurvival

from pymoo.util.display import disp_single_objective


class SingleObjectiveGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, var_type, **kwargs):
        set_if_none(kwargs, 'survival', FitnessSurvival())
        set_default_if_none(var_type, kwargs)
        super().__init__(**kwargs)
        self.func_display_attrs = disp_single_objective