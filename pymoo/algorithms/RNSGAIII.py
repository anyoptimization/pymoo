from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.rnsga_reference_line_survival import ReferenceLineSurvival

class RNSGAIII(GeneticAlgorithm):
    def __init__(self, var_type, ref_points=None, mu=0.1, ref_pop_size=None, method='uniform', p=None, **kwargs):
        """
        Algorithm Class for R-NSGA-III
        :param var_type: 'real'
        :param ref_points: numpy array of reference points
        :param mu: Shrink factor
        :param ref_pop_size: If the stuctured reference lines should be based off of a different population size
        than the actual population size. Default value is pop size.
        :param method: Reference direction generation method. Currently only 'uniform' or 'random'.
        :param p: If the structured reference directions should be based off of p gaps specify a p value, ortherwise
        reference directions will be based on the population size.
        :param kwargs: All other genetic operators.
        """
        self.ref_points = ref_points
        self.ref_dirs = None
        self.mu = mu
        self.method = method
        set_default_if_none(var_type, kwargs)
        set_if_none(kwargs, 'survival', None)
        self.ref_pop_size = ref_pop_size
        self.p = p
        super().__init__(**kwargs)

    def _initialize(self, problem):
        super()._initialize(problem)
        if self.ref_pop_size is None:
            self.ref_pop_size = self.pop_size
            # if survival not define differently
        if self.survival is None:
            self.survival = ReferenceLineSurvival(self.ref_dirs, problem.n_obj)

