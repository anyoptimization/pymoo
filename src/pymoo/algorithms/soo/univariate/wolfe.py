import numpy as np

from pymoo.algorithms.base.line import LineSearchProblem
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.solution import SolutionSet


class WolfeSearch(Algorithm):

    def __init__(self, c1=1e-4, c2=0.9, max_iter=10, **kwargs):

        super().__init__(**kwargs)
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

    def _setup(self, problem, **kwargs):
        assert isinstance(problem,
                          LineSearchProblem), "The wolfe search only purpose is to solve a line search problem!"
        self.pop = SolutionSet.create(problem.point)
        self.opt = self.pop

    def _set_optimum(self, force=False):
        pass

    def step(self):
        sol = self._infill()

        self.opt = sol
        self.termination.force_termination = True

    def _infill(self):

        problem, evaluator = self.problem, self.evaluator
        evaluator.skip_already_evaluated = False

        sol, direction = self.problem.point, self.problem.direction

        # the function value and gradient of the initial solution
        sol.set("alpha", 0.0)
        sol_F, sol_dF = sol.F[0], sol.get("dF")[0]

        def zoom(alpha_low, alpha_high, max_iter=100):

            while True:

                _alpha = (alpha_high.get("alpha") + alpha_low.get("alpha")) / 2
                _point = Individual(X=_alpha)
                evaluator.eval(problem, _point, evaluate_values_of=["F", "CV"])

                if _point.F[0] > sol_F + self.c1 * _alpha * sol_dF @ direction or _point.F[0] > alpha_low.F[0]:
                    alpha_high = _point
                else:
                    evaluator.eval(problem, _point, evaluate_values_of=["dF"])
                    point_dF = _point.get("dF")[0]

                    if np.abs(point_dF @ direction) <= -self.c2 * sol_dF @ direction:
                        return _point

                    if (point_dF @ direction) * (alpha_high.get("alpha") - alpha_low.get("alpha")) >= 0:
                        alpha_high = alpha_low

                    alpha_low = _point

        last = sol

        alpha = 1.0
        current = Individual(X=alpha)

        for i in range(1, self.max_iter + 1):

            # evaluate the solutions
            evaluator.eval(problem, current, evaluate_values_of=["F", "CV"])

            # get the values from the solution to be used to evaluate the conditions
            F, dF, _F = last.F[0], last.get("dF")[0], current.F[0]

            # if the wolfe condition is violate we have found our upper bound
            if _F > sol_F + self.c1 * sol_dF @ direction or (i > 1 and F >= _F):
                return zoom(last, current)

            # for the other condition we need the gradient information
            evaluator.eval(problem, current, evaluate_values_of=["dF"])
            _dF = current.get("dF")[0]

            if np.abs(_dF @ direction) <= -self.c2 * sol_dF @ direction:
                return current

            if _dF @ direction >= 0:
                return zoom(current, last)

            alpha = 2 * alpha
            last = current
            current = Individual(X=alpha)

        return current


def wolfe_line_search(problem, sol, direction, c1=1e-4, c2=0.9, max_iter=10, evaluator=None):
    # initialize the evaluator to be used (this will make sure evaluations are counted)
    evaluator = evaluator if evaluator is not None else Evaluator()
    evaluator.skip_already_evaluated = False

    # the function value and gradient of the initial solution
    sol.set("alpha", 0.0)
    sol_F, sol_dF = sol.F[0], sol.get("dF")[0]

    def zoom(alpha_low, alpha_high, max_iter=100):

        while True:

            _alpha = (alpha_high.get("alpha") + alpha_low.get("alpha")) / 2
            _point = Individual(X=sol.X + _alpha * direction, alpha=_alpha)
            evaluator.eval(problem, _point, evaluate_values_of=["F", "CV"])

            if _point.F[0] > sol_F + c1 * _alpha * sol_dF @ direction or _point.F[0] > alpha_low.F[0]:
                alpha_high = _point
            else:
                evaluator.eval(problem, _point, evaluate_values_of=["dF"])
                point_dF = _point.get("dF")[0]

                if np.abs(point_dF @ direction) <= -c2 * sol_dF @ direction:
                    return _point

                if (point_dF @ direction) * (alpha_high.get("alpha") - alpha_low.get("alpha")) >= 0:
                    alpha_high = alpha_low

                alpha_low = _point

    last = sol

    alpha = 1.0
    current = Individual(X=sol.X + alpha * direction, alpha=alpha)

    for i in range(1, max_iter + 1):

        # evaluate the solutions
        evaluator.eval(problem, current, evaluate_values_of=["F", "CV"])

        # get the values from the solution to be used to evaluate the conditions
        F, dF, _F = last.F[0], last.get("dF")[0], current.F[0]

        # if the wolfe condition is violate we have found our upper bound
        if _F > sol_F + c1 * sol_dF @ direction or (i > 1 and F >= _F):
            return zoom(last, current)

        # for the other condition we need the gradient information
        evaluator.eval(problem, current, evaluate_values_of=["dF"])
        _dF = current.get("dF")[0]

        if np.abs(_dF @ direction) <= -c2 * sol_dF @ direction:
            return current

        if _dF @ direction >= 0:
            return zoom(current, last)

        alpha = 2 * alpha
        last = current
        current = Individual(X=sol.X + alpha * direction, alpha=alpha)

    return current
