import numpy as np
from scipy.optimize import minimize as _minimize

from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.util.display import disp_single_objective
from pymoo.util.misc import at_least_2d_array


# =========================================================================================================
# Implementation
# =========================================================================================================


class NelderAndMead(Algorithm):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.func_display_attrs = disp_single_objective

    def _solve(self, problem, termination):

        self.pop = Population()
        self.n_gen = 0

        def fun(x):
            self.n_gen += 1

            ind = Individual()
            ind.X = x
            self.evaluator.eval(self.problem, ind, algorithm=self)
            self.pop = self.pop.merge([ind])

            self._each_iteration(self, first=self.n_gen == 1)

            return ind.F

        x0 = RandomSampling().sample(problem, Population(), 1).get("X")[0]

        res = _minimize(fun,
                        x0,
                        args=(),
                        method='Nelder-Mead',
                        bounds=None,
                        tol=None,
                        callback=None,
                        options={})

        self.pop = Population(1).set("X", at_least_2d_array(res.x),
                                     "F", at_least_2d_array(res.fun),
                                     "feasible", np.array([[True]]))

        return self.pop


def parse_bounds(problem):
    bounds = []
    for k in range(len(problem.xl)):
        bounds.append((problem.xl[k], problem.xu[k]))
    return bounds
