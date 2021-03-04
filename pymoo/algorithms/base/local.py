import abc

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.model.algorithm import Algorithm
from pymoo.model.population import pop_from_array_or_individual
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
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

        # if the local algorithm has not been initialized yet
        if not self.is_local_initialized:
            infills = self._local_initialize_infill()

            # the the algorithm does not implement the local initialization
            if infills is None:
                self.is_local_initialized = True
                return self._local_infill()
            else:
                return infills

        else:
            return self._local_infill()

    def _advance(self, infills=None, **kwargs):
        if not self.is_local_initialized:
            self._local_initialize_advance(infills=infills, **kwargs)
            self.is_local_initialized = True
        else:
            self._local_advance(infills=infills, **kwargs)

    def _local_initialize_infill(self):
        pass

    def _local_initialize_advance(self):
        pass

    @abc.abstractmethod
    def _local_infill(self):
        pass

    @abc.abstractmethod
    def _local_advance(self, **kwargs):
        pass
