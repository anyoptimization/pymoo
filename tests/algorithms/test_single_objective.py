import numpy as np
import pytest

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.optimize import minimize
from pymoo.problems.single import Sphere, Problem

SEEDS = np.arange(2, 5)


class SphereNoBounds(Sphere):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        self.xl = None
        self.xu = None


class SphereWithConstraints(Problem):

    def __init__(self, n_var=10):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=1, xl=-0, xu=1, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(x - 0.5), axis=1)
        out["G"] = 0.1 - out["F"]

    def _calc_pareto_front(self):
        return 0.1


def run(problem, algorithm, seed=1):
    res = minimize(problem, algorithm, seed=int(seed))
    f = res.F[0]
    print(res.CV)
    f_opt = problem.pareto_front()[0, 0]
    return f, f_opt


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('algorithm', [NelderMead(), PatternSearch(), PSO(), GA(), DE()],
                         ids=["nelder", "pattern", "pso", "ga", "de"])
def test_sphere(algorithm, seed):
    problem = Sphere()
    f, f_opt = run(problem, algorithm, seed=seed)
    np.testing.assert_almost_equal(f, f_opt, decimal=5)


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('algorithm', [NelderMead(), PatternSearch(), PSO(), GA(), DE()],
                         ids=["nelder", "pattern", "pso", "ga", "de"])
def test_sphere_with_constraints(algorithm, seed):
    problem = SphereWithConstraints()
    f, f_opt = run(problem, algorithm)
    np.testing.assert_almost_equal(f, f_opt, decimal=4)


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('clazz', [NelderMead, PatternSearch], ids=["nelder", "pattern"])
def test_sphere_no_bounds(clazz, seed):
    np.random.seed(seed)

    problem = SphereNoBounds()
    x0 = np.random.random(problem.n_var)

    algorithm = clazz(x0=x0)
    f, f_opt = run(problem, algorithm)
    np.testing.assert_almost_equal(f, f_opt, decimal=5)
    print(problem.__class__.__name__, algorithm.__class__.__name__, "Yes")
