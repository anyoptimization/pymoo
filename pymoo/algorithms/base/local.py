from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.algorithm import Algorithm, LoopwiseAlgorithm
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.util.display.single import SingleObjectiveOutput


class LocalSearch(LoopwiseAlgorithm, Algorithm):

    def __init__(self,
                 initial=None,
                 sampling=LatinHypercubeSampling(),
                 output=SingleObjectiveOutput(),
                 n_sample_points=20,
                 **kwargs):
        super().__init__(output=output, **kwargs)

        # the default termination if not specified otherwise
        self.termination = RobustTermination(SingleObjectiveSpaceTermination(tol=1e-8), period=10)

        # the type of initial sampling
        initial = initial if "x0" not in kwargs else kwargs["x0"]
        if initial is not None:
            sampling = initial

        self.initialization = Initialization(sampling)

        # the number of sampling points to determine x0
        self.n_sample_points = n_sample_points

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.n_sample_points, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        self.x0 = FitnessSurvival().do(self.problem, infills, n_survive=1, algorithm=self)[0]


