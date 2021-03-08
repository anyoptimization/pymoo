from pymoo.model.algorithm import Algorithm
from pymoo.model.population import pop_from_array_or_individual
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling


class LocalSearch(Algorithm):

    def __init__(self,
                 x0=None,
                 sampling=LatinHypercubeSampling(),
                 n_sample_points="auto",
                 **kwargs):

        super().__init__(**kwargs)
        self.x0 = x0
        self.sampling = sampling
        self.n_sample_points = n_sample_points

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)

        if self.n_sample_points == "auto":
            self.n_sample_points = self.problem.n_var * 5

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)

        # no initial point is provided - sample in bounds and take the best
        if self.x0 is None:
            if not self.problem.has_bounds():
                raise Exception("Either provide an x0 or a problem with variable bounds!")

            self.pop = self.sampling.do(self.problem, self.n_sample_points)
        else:
            self.pop = pop_from_array_or_individual(self.x0)

        self.evaluator.eval(self.problem, self.pop, algorithm=self)
        self._set_optimum()
        if self.opt is not None and len(self.opt) > 0:
            self.x0 = self.opt[0]
