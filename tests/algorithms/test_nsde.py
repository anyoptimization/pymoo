import pytest

from pymoo.algorithms.moo.nsde import NSDE
from pymoo.optimize import minimize
from pymoo.problems import get_problem


@pytest.mark.parametrize("selection", ["rand", "best", "current-to-best", "current-to-rand", "ranked"])
@pytest.mark.parametrize("crossover", ["bin", "exp"])
@pytest.mark.parametrize("de_repair", ["bounce-back", "midway", "rand-init", "to-bounds"])
def test_nsde_run(selection, crossover, de_repair):
    problem = get_problem("zdt1")
    alg = NSDE(pop_size=30, variant=f"DE/{selection}/1/{crossover}", CR=0.5, F=(0.3, 1.0),
                de_repair=de_repair)
    res = minimize(problem, alg, ("n_gen", 20), seed=1, verbose=False)
    assert len(res.opt) > 0


def test_nsde_perf():
    from pymoo.indicators.igd import IGD
    problem = get_problem("zdt1")
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    alg = NSDE(pop_size=100, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back")
    res = minimize(problem, alg, ("n_gen", 200), seed=1, verbose=False)
    assert igd.do(res.F) <= 0.05
