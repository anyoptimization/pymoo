import abc

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import pop_from_array_or_individual
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination


class LocalSearch(Algorithm):

    def __init__(self,
                 x0=None,
                 sampling=LatinHypercubeSampling(),
                 display=SingleObjectiveDisplay(),
                 n_sample_points="auto",
                 n_max_sample_points=20,
                 **kwargs):

        super().__init__(display=display, **kwargs)
        self.sampling = sampling
        self.n_sample_points = n_sample_points
        self.n_max_sample_points = n_max_sample_points
        self.x0 = x0
        self.default_termination = SingleObjectiveSpaceToleranceTermination(n_last=5, tol=1e-8)

        self.is_local_initialized = False

    def _setup(self, problem, x0=None, **kwargs):
        if self.x0 is None:
            self.x0 = x0

    def _initialize_infill(self):
        # calculate the default number of sample points
        if self.n_sample_points == "auto":
            self.n_sample_points = min(self.problem.n_var * 5, self.n_max_sample_points)

        # no initial point is provided - sample in bounds and take the best
        if self.x0 is None:
            if not self.problem.has_bounds():
                raise Exception("Either provide an x0 or a problem with variable bounds!")
            pop = self.sampling.do(self.problem, self.n_sample_points)
        else:
            pop = pop_from_array_or_individual(self.x0)

        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)

        self.evaluator.eval(self.problem, infills, algorithm=self)
        self.x0 = FitnessSurvival().do(self.problem, infills, n_survive=1)[0]

    def _infill(self):
        if not self.is_local_initialized:
            return self._local_initialize_infill()
        else:
            return self._local_infill()

    def _advance(self, **kwargs):
        if not self.is_local_initialized:
            self.is_local_initialized = True
            return self._local_initialize_advance(**kwargs)
        else:
            return self._local_advance(**kwargs)

    def _local_initialize_infill(self, *args, **kwargs):
        return self._local_infill(*args, **kwargs)

    def _local_initialize_advance(self, *args, **kwargs):
        return self._local_advance(*args, **kwargs)

    @abc.abstractmethod
    def _local_infill(self):
        pass

    @abc.abstractmethod
    def _local_advance(self, **kwargs):
        pass
