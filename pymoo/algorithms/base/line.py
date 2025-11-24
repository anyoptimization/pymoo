import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual
from pymoo.core.meta import Meta
from pymoo.core.problem import Problem
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside


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

    def _initialize_infill(self):

        # x could be a vector or an individual
        if isinstance(self.point, np.ndarray):
            self.point = Individual(X=self.point)

        # make sure it is evaluated - if not yet also get the gradient
        if self.point.get("F") is None:
            self.evaluator.eval(self.problem, self.point, algorithm=self)

        self.infill = self.point


class LineSearchProblem(Meta, Problem):

    def __init__(self, problem, point, direction, strict_bounds=True, xl=0.0, xu=np.inf):
        super().__init__(problem)
        self.n_var = 1
        self.xl, self.xu = np.array([xl]), np.array([xu])

        self.point = point
        self.direction = direction
        self.strict_bounds = strict_bounds

    def _evaluate(self, alpha, out, *args, **kwargs):
        out["alpha"] = alpha

        x = self.point.X + alpha * self.direction
        if self.strict_bounds:
            x = set_to_bounds_if_outside(x, self.xl, self.xu)
        out["__X__"] = x

        super()._evaluate(x, out, *args, **kwargs)
