from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.rnsga_reference_line_survival import ReferenceLineSurvival
from pymoo.util.reference_directions import get_ref_dirs_from_points
import numpy as np

class RNSGAIII(GeneticAlgorithm):
    def __init__(self, var_type, ref_dirs=None, ep=0.001, **kwargs):
        self.ref_dirs = np.array(ref_dirs)
        self.epsilon = ep
        set_default_if_none(var_type, kwargs)
        set_if_none(kwargs, 'survival', None)
        super().__init__(**kwargs)

    def _initialize(self, problem):
        super()._initialize(problem)

        # if survival not define differently
        if self.survival is None:

            # set the survival method itself
            self.survival = ReferenceLineSurvival(self.ref_dirs, self.epsilon)
