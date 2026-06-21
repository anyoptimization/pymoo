import pytest

from pymoo.algorithms.moo.gde3 import GDE3, GDE3MNN, GDE32NN, GDE3PCD
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem


@pytest.mark.parametrize("selection", ["rand", "current-to-rand", "ranked"])
@pytest.mark.parametrize("crossover", ["bin", "exp"])
def test_gde3_run(selection, crossover):
    problem = get_problem("zdt1")
    alg = GDE3(pop_size=30, variant=f"DE/{selection}/1/{crossover}", CR=0.5, F=(0.3, 1.0))
    res = minimize(problem, alg, ("n_gen", 20), seed=1, verbose=False)
    assert len(res.opt) > 0


@pytest.mark.parametrize("cls", [GDE3MNN, GDE32NN, GDE3PCD])
def test_gde3_variants_run(cls):
    problem = get_problem("zdt1")
    alg = cls(pop_size=30, variant="DE/rand/1/bin", CR=0.5, F=(0.3, 1.0))
    res = minimize(problem, alg, ("n_gen", 20), seed=1, verbose=False)
    assert len(res.opt) > 0


def test_gde3_perf():
    problem = get_problem("zdt1")
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    alg = GDE3(pop_size=100, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back")
    res = minimize(problem, alg, ("n_gen", 200), seed=1, verbose=False)
    assert igd.do(res.F) <= 0.05
