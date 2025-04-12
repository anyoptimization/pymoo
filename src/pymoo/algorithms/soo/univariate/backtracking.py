from pymoo.algorithms.base.line import LineSearch
from pymoo.core.evaluator import Evaluator
from pymoo.core.solution import Solution, SolutionSet
from pymoo.core.termination import NoTermination
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere


class BacktrackingLineSearch(LineSearch):

    def __init__(self, alpha=0.1, beta=0.33, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.termination = NoTermination()

    def _initialize_infill(self):
        super()._initialize_infill()
        self.t = 1.0
        self.point.set("t", self.t)

        # TODO:  evaluate_values_of needs to be moved into evaluator to be more flexible
        if self.point.get("dF") is None:
            self.evaluator.eval(self.problem, self.point, evaluate_values_of=["dF"], algorithm=self)

    def step(self):
        t = self.t
        x, f, df, p = self.point.X, self.point.F[0], self.point.get("dF")[0], self.direction

        infill = Solution(X=x + t * p, t=t)
        self.evaluator.eval(self.problem, infill, algorithm=self, evaluate_values_of=["F"])
        _f = infill.F[0]

        self.pop = SolutionSet.merge(self.pop, infill)
        self.infill = infill

        if _f < f + self.alpha * self.t * df.T @ p or self.t <= 1e-8:
            self.termination.force_termination = True
        else:
            self.t = self.t * self.beta


if __name__ == '__main__':
    import numpy as np

    problem = Sphere()

    X = np.array(np.random.random(problem.n_var))

    point = Solution(X=X)
    Evaluator(evaluate_values_of=["F", "dF"]).eval(problem, point)

    direction = - point.get("dF")[0]

    algorithm = BacktrackingLineSearch().setup(problem, point=point, direction=direction)._initialize()

    res = minimize(problem, algorithm)

    print(res.X)
