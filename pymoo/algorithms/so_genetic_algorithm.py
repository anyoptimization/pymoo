from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none


class SingleObjectiveGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, var_type, **kwargs):
        set_default_if_none(var_type, **kwargs)
        super().__init__(**kwargs)
