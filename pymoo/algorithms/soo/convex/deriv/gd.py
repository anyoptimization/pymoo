import numpy as np

from pymoo.algorithms.base.gradient import GradientBasedAlgorithm
from pymoo.factory import Sphere
from pymoo.model.population import Population
from pymoo.optimize import minimize
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination


class GradientDescent(GradientBasedAlgorithm):

    def __init__(self, damped=True, **kwargs):
        super().__init__(**kwargs)
        self.damped = damped
        self.dF = None
        self.default_termination = SingleObjectiveSpaceToleranceTermination()

    def direction(self):
        return - self.dF

    def _next(self):
        sol = self.opt[0]
        self.dF = self.gradient(sol)

        _next = self.inexact_line_search(sol, self.direction())
        self.pop = Population.merge(self.opt, _next)

        if np.linalg.norm(self.dF) ** 2 < 1e-8:
            self.termination.force_termination = True


class CoordinateDescent(GradientDescent):

    def direction(self):
        dF = self.dF
        p = np.zeros(len(dF))
        k = np.abs(dF).argmax()
        p[k] = 1 * np.sign(dF[k])
        return - p


problem = Sphere(n_var=8)

algorithm = CoordinateDescent()

res = minimize(problem, algorithm, verbose=True, seed=1)
