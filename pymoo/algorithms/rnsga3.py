from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.survival.rnsga_reference_line_survival import ReferenceLineSurvival


class RNSGAIII(GeneticAlgorithm):
    def __init__(self, var_type, ref_points=None, mu=0.1, ref_pop_size=None, method='uniform', p=None, **kwargs):
        """

        Parameters
        ----------
        var_type : string
            Variable type which must be real in this case

        ref_points : numpy.array
            Reference points to be focused on during the evolutionary computation.

        mu : double
            The shrink factor

        ref_pop_size : int
            If the structured reference lines should be based off of a different population size
            than the actual population size. Default value is pop size.

        p : double
            If the structured reference directions should be based off of p gaps specify a p value, otherwise
            reference directions will be based on the population size.

        ref_sampling_method : string
            Reference direction generation method. Currently only 'uniform' or 'random'.

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

        if self.survival is None:
            self.survival = ReferenceLineSurvival(self.ref_dirs, problem.n_obj)
