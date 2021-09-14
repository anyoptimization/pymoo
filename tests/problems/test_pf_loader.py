import numpy as np

from pymoo.problems.many import DTLZ6
from pymoo.problems.multi import ZDT1, Kursawe, OSY, CTP1
from pymoo.util.cache import Cache


def my_loader():
    return np.arange(10)


def test_load_func():
    loader = Cache(my_loader)
    pf = loader.exec()
    assert len(pf) == 10


def test_load_cache():
    loader = Cache(my_loader)
    assert id(loader.exec()) == id(loader.exec())


def test_load_functional():
    problem = ZDT1()
    pf = problem.pareto_front(n_pareto_points=200)
    assert len(pf) == 200
    assert id(pf) == id(problem.pareto_front())
    assert id(pf) != id(problem.pareto_front(use_cache=False))


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
