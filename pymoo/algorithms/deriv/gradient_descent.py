import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.model.population import Population
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


class GradientBasedAlgorithm(Algorithm):

    def __init__(self,
                 X,
                 dX=None,
                 objective=0,
                 display=SingleObjectiveDisplay(),
                 **kwargs) -> None:
        super().__init__(display=display, **kwargs)

        self.objective = objective
        self.n_restarts = 0
        self.default_termination = SingleObjectiveDefaultTermination()

        self.X, self.dX = X, dX
        self.F, self.CV = None, None

        if self.X.ndim == 1:
            self.X = np.atleast_2d(X)

    def _initialize(self):
        self._next()

    def _next(self):

        # create a copy from the current values - if restart is necessary
        _X = np.copy(self.X)

        # if the gradient was not provided yet evaluate it
        if self.F is None or self.dX is None:
            # evaluate the problem and get the information of gradients
            F, dX, CV = self.problem.evaluate(self.X, return_values_of=["F", "dF", "CV"])

            # because we only consider one objective here
            F = F[:, [self.objective]]
            dX = dX[:, self.objective]

            # increase the evaluation count
            self.evaluator.n_eval += len(self.X)

        has_improved = self.F is None or np.any(F < self.F)
        is_gradient_valid = np.all(~np.isnan(dX))

        # if the gradient did lead to an improvement
        if has_improved:

            self.F, self.dX, self.CV = F, dX, CV

            # if the gradient is valid and has no nan values
            if is_gradient_valid:

                # make the step and check out of bounds for X
                self.apply()
                self.X = set_to_bounds_if_outside_by_problem(self.problem, self.X)

                # set the population object for automatic print
                self.pop = Population(len(self.X)).set("X", self.X, "F", self.F,
                                                       "CV", self.CV, "feasible", self.CV <= 0)

            # otherwise end the termination form now on
            else:
                print("WARNING: GRADIENT ERROR", self.dX)
                self.termination.force_termination = True

        # otherwise do a restart of the algorithm
        else:
            self.X = _X
            self.restart()
            self.n_restarts += 1

        # set the gradient to none to be ready for the next iteration
        self.dX = None


class GradientDescent(GradientBasedAlgorithm):

    def __init__(self, X, learning_rate=0.005, **kwargs) -> None:
        super().__init__(X, **kwargs)
        self.learning_rate = learning_rate

    def restart(self):
        self.learning_rate /= 2

    def apply(self):
        self.X = self.X - self.learning_rate * self.dX
