import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.reference_point_survival import ReferencePointSurvival


class ERNSGA2(GeneticAlgorithm):

    def __init__(self,
                 var_type,
                 ref_points=None,
                 epsilon=0.001,
                 weights=None,
                 normalize=True,
                 **kwargs):
        self.ref_dirs = np.array(ref_points)
        self.epsilon = epsilon
        self.weights = weights
        self.normalize = normalize
        set_default_if_none(var_type, kwargs)
        set_if_none(kwargs, 'survival', None)
        super().__init__(**kwargs)

    def _initialize(self, problem):
        super()._initialize(problem)

        if self.survival is None:
            self.survival = ReferencePointSurvival(self.ref_dirs, self.epsilon, self.weights, problem.n_obj)
