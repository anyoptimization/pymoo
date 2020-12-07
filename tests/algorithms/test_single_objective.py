import unittest

import numpy as np

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.so_nelder_mead import NelderMead
from pymoo.algorithms.so_pattern_search import PatternSearch
from pymoo.algorithms.so_pso import PSO
from pymoo.factory import Sphere, Problem
from pymoo.optimize import minimize


class SphereNoBounds(Sphere):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        self.xl = None
        self.xu = None


class SphereWithConstraints(Problem):

    def __init__(self, n_var=10):
        super().__init__(n_var=n_var, n_obj=1, n_constr=1, xl=-0, xu=1, type_var=np.double)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(x - 0.5), axis=1)
        out["G"] = 0.1 - out["F"]

    def _calc_pareto_front(self):
        return 0.1


def test(problem, algorithm):
    res = minimize(problem, algorithm, seed=1)
    f = res.F[0]
    print(res.CV)
    f_opt = problem.pareto_front()[0, 0]
    return f, f_opt


class SingleObjectiveAlgorithmTest(unittest.TestCase):

    def test_sphere(self):
        problem = Sphere()
        for algorithm in [NelderMead(), PatternSearch(), PSO(), GA()]:
            f, f_opt = test(problem, algorithm)
            self.assertAlmostEqual(f, f_opt, places=5)
            print(problem.__class__.__name__, algorithm.__class__.__name__, "Yes")

    def test_sphere_with_constraints(self):
        problem = SphereWithConstraints()
        for algorithm in [GA(), NelderMead(), PatternSearch()]:
            f, f_opt = test(problem, algorithm)
            self.assertAlmostEqual(f, f_opt, places=3)
            print(problem.__class__.__name__, algorithm.__class__.__name__, "Yes")

    def test_sphere_no_bounds(self):
        problem = SphereNoBounds()
        x0 = np.random.random(problem.n_var)

        for algorithm in [NelderMead(x0=x0), PatternSearch(x0=x0)]:
            f, f_opt = test(problem, algorithm)
            self.assertAlmostEqual(f, f_opt, places=5)
            print(problem.__class__.__name__, algorithm.__class__.__name__, "Yes")


if __name__ == '__main__':
    unittest.main()
