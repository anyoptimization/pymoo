import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.problem import MetaProblem


class LineSearch(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.point, self.direction = None, None

    def setup(self, problem, point=None, direction=None, **kwargs):
        super().setup(problem, **kwargs)

        msg = "Only problems with one objective and no constraints can be solved using a line search!"
        assert not problem.has_constraints() and problem.n_obj == 1, msg

        assert point is not None, "You have to define a starting point for the algorithm"
        self.point = point

        assert direction is not None, "You have to define a direction point for the algorithm"
        self.direction = direction

        return self

    def _initialize(self):

        # x could be a vector or an individual
        if isinstance(self.point, np.ndarray):
            self.point = Individual(X=self.point)

        # make sure it is evaluated - if not yet also get the gradient
        if self.point.get("F") is None:
            self.evaluator.eval(self.problem, self.point, algorithm=self)

        self.infill = self.point


class LineSearchProblem(MetaProblem):

    def __init__(self, problem, point, direction, xl=0.0, xu=1.0):
        super().__init__(problem)
        self.n_var = 1
        self.xl = np.array([xl])
        self.xu = np.array([xu])
        self.point = point
        self.direction = direction

    def _evaluate(self, x, out, *args, **kwargs):
        _x = self.point + x * self.direction
        out["__X__"] = _x
        super()._evaluate(_x, out, *args, **kwargs)
