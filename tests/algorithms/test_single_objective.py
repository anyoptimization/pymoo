import numpy as np

from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
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


def run(problem, algorithm):
    res = minimize(problem, algorithm, seed=1)
    f = res.F[0]
    print(res.CV)
    f_opt = problem.pareto_front()[0, 0]
    return f, f_opt


def test_sphere():
    problem = Sphere()
    for algorithm in [NelderMead(), PatternSearch(), PSO(), GA()]:
        f, f_opt = run(problem, algorithm)
        np.testing.assert_almost_equal(f, f_opt, decimal=5)
        print(problem.__class__.__name__, algorithm.__class__.__name__, "Yes")


def test_sphere_with_constraints():
    problem = SphereWithConstraints()
    for algorithm in [GA(), NelderMead(), PatternSearch()]:
        f, f_opt = run(problem, algorithm)
        np.testing.assert_almost_equal(f, f_opt, decimal=5)
        print(problem.__class__.__name__, algorithm.__class__.__name__, "Yes")


def test_sphere_no_bounds():
    problem = SphereNoBounds()
    x0 = np.random.random(problem.n_var)

    for algorithm in [NelderMead(x0=x0), PatternSearch(x0=x0)]:
        f, f_opt = run(problem, algorithm)
        np.testing.assert_almost_equal(f, f_opt, decimal=5)
        print(problem.__class__.__name__, algorithm.__class__.__name__, "Yes")
