import numpy as np
import pytest

from pymoo.problems import get_problem
from pymoo.problems.many import ZCAT1, ZCAT14, ZCAT20


def test_zcat_default_settings():
    problem = ZCAT1()
    assert problem.n_var == 30
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 0
    assert problem.xl[0] == -0.5
    assert problem.xu[0] == 0.5
    assert problem.xl[-1] == -15.0
    assert problem.xu[-1] == 15.0


def test_zcat_evaluate_produces_finite_objectives():
    problem = get_problem("zcat3", n_var=10, n_obj=3)
    x_mid = (problem.xl + problem.xu) / 2.0
    x = np.tile(x_mid, (4, 1))

    f = problem.evaluate(x)

    assert f.shape == (4, 3)
    assert np.all(np.isfinite(f))


def test_zcat_one_dimensional_pareto_set_problem_allows_n_var_one():
    problem = ZCAT14(n_var=1, n_obj=3)
    f = problem.evaluate(np.array([[0.0]]))

    assert f.shape == (1, 3)
    assert np.all(np.isfinite(f))


def test_zcat_rejects_invalid_number_of_variables():
    with pytest.raises(ValueError):
        ZCAT1(n_var=1, n_obj=3)


def test_zcat_get_problem_and_name():
    problem = get_problem("zcat20")
    assert isinstance(problem, ZCAT20)
    assert problem.name() == "ZCAT20"


def test_zcat_reference_front_loading_for_2d_and_3d():
    pf2 = ZCAT1(n_obj=2).pareto_front(use_cache=False)
    pf3 = ZCAT1(n_obj=3).pareto_front(use_cache=False)

    assert pf2.shape[1] == 2
    assert pf3.shape[1] == 3
    assert len(pf2) == 1000
    assert len(pf3) == 1000
