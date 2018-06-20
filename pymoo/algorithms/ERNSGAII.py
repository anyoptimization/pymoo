from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.rnsga_reference_line_survival import ReferenceLineSurvival
import numpy as np

class ERNSGAII(GeneticAlgorithm):
    """
    Class Extended RNSGA-II
    """
    def __init__(self, var_type, ref_points=None, epsilon=0.001, weights=None, **kwargs):
        self.ref_dirs = np.array(ref_points)
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
            self.survival = ReferenceLineSurvival(self.ref_dirs, self.epsilon, self.weights)
