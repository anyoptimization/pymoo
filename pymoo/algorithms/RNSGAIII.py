from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.reference_point_survival import ReferencePointSurvival


class RNSGAIII(GeneticAlgorithm):
    def __init__(self, var_type, ref_points, epsilon=0.001, weights=None, **kwargs):
        self.ref_points = ref_points
        self.epsilon = epsilon
        self.weights = weights
        set_default_if_none(var_type, kwargs)
        set_if_none(kwargs, 'survival', None)
        super().__init__(**kwargs)

    def _initialize(self, problem):
        super()._initialize(problem)

        # if survival not define differently
        if self.survival is None:
            # set the survival method itself
            self.survival = ReferencePointSurvival(self.ref_points, self.epsilon, self.weights)