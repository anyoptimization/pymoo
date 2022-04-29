from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.algorithm import Algorithm, LoopwiseAlgorithm
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.util.display import SingleObjectiveDisplay


class LocalSearch(LoopwiseAlgorithm, Algorithm):

    def __init__(self,
                 sampling=LatinHypercubeSampling(),
                 display=SingleObjectiveDisplay(),
                 n_sample_points=20,
                 x0=None,
                 **kwargs):
        super().__init__(display=display, **kwargs)

        # the default termination if not specified otherwise
        self.termination = RobustTermination(SingleObjectiveSpaceTermination(tol=1e-8), n=10)

        # the type of initial sampling
        self.initialization = Initialization(sampling)

        # the number of sampling points to determine x0
        self.n_sample_points = n_sample_points

        # the starting point for the local search
        self.x0 = x0

    def _initialize_infill(self):
        if self.x0 is None:
            return self.initialization.do(self.problem, self.n_sample_points, algorithm=self)
        else:
            return self.infill()

    def _initialize_advance(self, infills=None, **kwargs):
        if self.x0 is None:
            self.x0 = FitnessSurvival().do(self.problem, infills, n_survive=1)[0]
        else:
            return self.advance(infills=infills, **kwargs)

