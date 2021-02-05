from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.model.algorithm import Algorithm
from pymoo.model.population import pop_from_array_or_individual, Population
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination


class LocalSearch(Algorithm):

    def __init__(self,
                 x0=None,
                 sampling=LatinHypercubeSampling(),
                 display=SingleObjectiveDisplay(),
                 n_sample_points="auto",
                 **kwargs):

        super().__init__(display=display, **kwargs)
        self.sampling = sampling
        self.n_sample_points = n_sample_points
        self.x0 = x0
        self.default_termination = SingleObjectiveSpaceToleranceTermination(n_last=5, tol=1e-8)

        if kwargs.get("x0") is not None:
            "The interface in pymoo 0.5.0 has changed. The x0 parameter has now to be provided to the minimize " \
            "method directly or while calling setup."

    def setup(self, problem, x0=None, **kwargs):
        super().setup(problem, **kwargs)

        # calculate the default number of sample points
        if self.n_sample_points == "auto":
            self.n_sample_points = min(self.problem.n_var * 5, 50)

        # no initial point is provided - sample in bounds and take the best
        if self.x0 is None:
            if not self.problem.has_bounds():
                raise Exception("Either provide an x0 or a problem with variable bounds!")
            pop = self.sampling.do(self.problem, self.n_sample_points)

        else:
            pop = pop_from_array_or_individual(self.x0)

        self.evaluator.eval(problem, pop, algorithm=self)
        self.x0 = FitnessSurvival().do(problem, pop, n_survive=1)[0]

        return self

    def _initialize(self):
        return Population.create(self.x0)
