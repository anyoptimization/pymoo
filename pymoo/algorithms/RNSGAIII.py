from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.reference_line_survival import ReferenceLineSurvival
from pymoo.util.reference_directions import get_ref_dirs_from_points

class RNSGAIII(GeneticAlgorithm):
    def __init__(self, var_type, ref_points=None, alpha=0.1, ref_pop_size=None, method='uniform', **kwargs):
        self.ref_points = ref_points
        self.ref_dirs = None
        self.alpha = alpha
        self.method = method
        set_default_if_none(var_type, kwargs)
        set_if_none(kwargs, 'survival', None)
        self.ref_pop_size = ref_pop_size
        super().__init__(**kwargs)

    def _initialize(self, problem):
        super()._initialize(problem)

        # if survival not define differently
        if self.survival is None:
            # if ref dirs are not initialized do it based on the population size
            if self.ref_dirs is None:
                if self.ref_pop_size is not None:
                    self.ref_dirs = get_ref_dirs_from_points(self.ref_points, problem.n_obj, self.ref_pop_size, alpha=self.alpha, method=self.method)
                else:
                    self.ref_dirs = get_ref_dirs_from_points(self.ref_points, problem.n_obj, self.pop_size, alpha=self.alpha, method=self.method)


            self.survival = ReferenceLineSurvival(self.ref_dirs)

