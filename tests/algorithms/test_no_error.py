import pytest

from pymoo.algorithms.moo.gde3 import GDE3
from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.soo.convex.nonderiv.cmaes import CMAES
from pymoo.algorithms.soo.convex.nonderiv.nelder_mead import NelderMead
from pymoo.algorithms.soo.convex.nonderiv.pattern_search import PatternSearch
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.direct import DIRECT
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.factory import Sphere, Rosenbrock, Himmelblau, ZDT1, ZDT2, ZDT3, get_reference_directions
from pymoo.optimize import minimize
import numpy as np

from pymoo.performance_indicator.igd import IGD

SINGLE_OBJECTIVE_PROBLEMS = [Sphere(n_var=10), Himmelblau(), Rosenbrock()]

SINGLE_OBJECTIVE_ALGORITHMS = [PatternSearch(), CMAES(), NelderMead(), DIRECT(), PSO(), DE()]


@pytest.mark.parametrize('problem', SINGLE_OBJECTIVE_PROBLEMS)
@pytest.mark.parametrize('algorithm', SINGLE_OBJECTIVE_ALGORITHMS)
def test_singe_obj(problem, algorithm):
    res = minimize(problem, algorithm, seed=1, verbose=True)
    fmin = problem.pareto_front().flatten()[0]
    np.testing.assert_almost_equal(fmin, res.F[0], decimal=3)


MULTI_OBJECTIVE_PROBLEMS = [ZDT1()]


ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
MULTI_OBJECTIVE_ALGORITHMS = [NSGA2(), GDE3(), RVEA(ref_dirs), MOEAD(ref_dirs), ParallelMOEAD(ref_dirs)]


@pytest.mark.parametrize('problem', MULTI_OBJECTIVE_PROBLEMS)
@pytest.mark.parametrize('algorithm', MULTI_OBJECTIVE_ALGORITHMS)
def test_multi_obj(problem, algorithm):
    res = minimize(problem, algorithm, ('n_gen', 300), seed=1, verbose=True)
    pf = problem.pareto_front()
    assert IGD(pf).do(res.F) < 0.05
