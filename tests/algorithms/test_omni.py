import numpy as np
import pytest

from pymoo.algorithms.moo.omni import OmniOptimizer
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.multi.omnitest import OmniTest


@pytest.mark.parametrize("problem_name", ["zdt1", "zdt2", "zdt3"])
def test_omni_standard_problems(problem_name):
    """OmniOptimizer should handle standard MOO benchmarks correctly."""
    problem = get_problem(problem_name)
    alg = OmniOptimizer(pop_size=50)
    res = minimize(problem, alg, ("n_gen", 50), seed=1, verbose=False)
    assert len(res.opt) > 0


def test_omni_omnitest():
    """OmniOptimizer should find all Pareto subsets in decision space."""
    problem = OmniTest(n_var=2)
    alg = OmniOptimizer(pop_size=200)
    res = minimize(problem, alg, ("n_gen", 300), seed=1, verbose=False)

    assert len(res.opt) > 0

    # OmniTest with n_var=2 has 3^2=9 Pareto-optimal subsets in X-space.
    # Check that solutions are spread across all of them.
    bands = [(1, 1.5), (3, 3.5), (5, 5.5)]
    found = set()
    for x in res.opt.get("X"):
        key = tuple(next((i for i, (lo, hi) in enumerate(bands) if lo <= xi <= hi), -1) for xi in x)
        if -1 not in key:
            found.add(key)
    assert len(found) == 9, f"Expected 9 Pareto subsets, found {len(found)}"


def test_omni_perf_zdt1():
    problem = get_problem("zdt1")
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    alg = OmniOptimizer(pop_size=100)
    res = minimize(problem, alg, ("n_gen", 200), seed=1, verbose=False)
    assert igd.do(res.F) <= 0.02


def test_omni_constrained():
    """OmniOptimizer should handle constrained problems (uses NSGA2 base)."""
    problem = get_problem("truss2d")
    alg = OmniOptimizer(pop_size=50)
    res = minimize(problem, alg, ("n_gen", 100), seed=1, verbose=False)
    assert len(res.opt) > 0
