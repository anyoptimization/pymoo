from pymoo.algorithms.nsga3 import NSGA3
from pymoo.operators.survival.rnsga_reference_line_survival import ReferenceLineSurvival


class RNSGA3(NSGA3):

    def __init__(self,
                 ref_points=None,
                 mu=0.1,
                 method="uniform",
                 ref_pop_size=None,
                 p=None,
                 **kwargs):

        """

        Parameters
        ----------

        ref_points : numpy.array
            Reference points to be focused on during the evolutionary computation.
        mu : double
            The shrink factor
        method : str
            Weight sampling method
        ref_pop_size : int
            If the structured reference lines should be based off of a different population size
            than the actual population size. Default value is pop size.
        p : double
            If the structured reference directions should be based off of p gaps specify a p value, otherwise
            reference directions will be based on the population size.
        """

        super().__init__(**kwargs)

        self.ref_points = ref_points
        self.method = method
        self.mu = mu
        self.ref_pop_size = ref_pop_size
        self.p = p

    def _initialize(self):
        pop = super()._initialize()

        if self.ref_pop_size is None:
            self.ref_pop_size = self.pop_size

        self.survival = ReferenceLineSurvival(self.ref_dirs, self.ref_points, self.ref_pop_size, self.mu, self.method, self.p, self.problem.n_obj)
        return pop
