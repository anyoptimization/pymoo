
from pymoo.problems.many import DTLZ6, DTLZ1
from pymoo.problems.multi import ZDT1, Kursawe, OSY, CTP1
from pymoo.util.ref_dirs import get_reference_directions


def test_load_functional():
    problem = ZDT1()
    pf = problem.pareto_front(n_pareto_points=200)
    assert len(pf) == 200
    assert id(pf) == id(problem.pareto_front())


def test_load_functional_with_param():
    problem = DTLZ1()
    ref_dirs1 = get_reference_directions("uniform", 3, n_partitions=13)
    ref_dirs2 = get_reference_directions("uniform", 3, n_partitions=12)

    pf = problem.pareto_front(ref_dirs=ref_dirs1)
    assert len(pf) == 105

    pf = problem.pareto_front(use_cache=True, ref_dirs=ref_dirs2)
    assert len(pf) == 105

    pf = problem.pareto_front(use_cache=False, ref_dirs=ref_dirs2)
    assert len(pf) == 91


def test_load_remote_kur():
    problem = Kursawe()
    pf = problem.pareto_front()
    assert len(pf) == 100


def test_load_remote_osy():
    problem = OSY()
    pf = problem.pareto_front()
    assert len(pf) == 99


def test_load_remote_dtlz6():
    problem = DTLZ6()
    pf = problem.pareto_front()
    assert len(pf) == 10201


def test_load_remote_ctp():
    problem = CTP1()
    pf = problem.pareto_front()
    assert len(pf) == 1000
