from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.fitness_survival import FitnessSurvival


class SingleObjectiveGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, var_type, **kwargs):
        set_default_if_none(var_type, kwargs)
        set_if_none(kwargs, 'survival', FitnessSurvival())
        super().__init__(**kwargs)
