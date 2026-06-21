import pytest

from pymoo.algorithms.moo.nsder import NSDER
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions


@pytest.mark.parametrize("selection", ["rand", "current-to-rand", "ranked"])
@pytest.mark.parametrize("crossover", ["bin", "exp"])
def test_nsder_run(selection, crossover):
    problem = get_problem("dtlz2")
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)
    alg = NSDER(ref_dirs=ref_dirs, variant=f"DE/{selection}/1/{crossover}", CR=0.5, F=(0.0, 1.0))
    res = minimize(problem, alg, ("n_gen", 20), seed=1, verbose=False)
    assert len(res.opt) > 0


def test_nsder_wrong_ref_dirs():
    problem = get_problem("zdt1")  # 2 objectives
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)  # 3 objectives
    alg = NSDER(ref_dirs=ref_dirs, variant="DE/rand/1/bin")
    with pytest.raises(ValueError, match="Dimensionality"):
        minimize(problem, alg, ("n_gen", 5), seed=1)


def test_nsder_perf():
    problem = get_problem("dtlz2")
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    alg = NSDER(ref_dirs=ref_dirs, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 1.0))
    res = minimize(problem, alg, ("n_gen", 200), seed=1, verbose=False)
    assert igd.do(res.F) <= 0.06
