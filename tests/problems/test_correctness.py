import numpy as np
import pytest

from pymoo.core.individual import calc_cv
from pymoo.problems import get_problem
from pymoo.util.misc import at_least_2d_array
from tests.test_util import load_to_test_resource

problems = [
    ('DTLZ1', [10, 3]), ('DTLZ2', [10, 3]), ('DTLZ3', [10, 3]), ('DTLZ4', [10, 3]), ('DTLZ5', [10, 3]),
    ('DTLZ6', [10, 3]), ('DTLZ7', [10, 3]),
    ('C1DTLZ1', [12, 3]), ('C2DTLZ2', [12, 3]), ('C3DTLZ1', [12, 3]), ('C3DTLZ4', [7, 3]),
    ('ZDT1', [10]), ('ZDT2', [10]), ('ZDT3', [10]), ('ZDT4', [10]), ('ZDT6', [10]),
    ('TNK', []), ('Rosenbrock', [10]), ('Rastrigin', [10]), ('Griewank', [10]), ('OSY', []), ('Kursawe', []),
    ('Welded_Beam', []), ('Carside', []), ('BNH', []),
    # ('ctp1', []), ('ctp2', []), ('ctp3', []), ('ctp4', []), ('ctp5', []), ('ctp6', []), ('ctp7', []), ('ctp8', []),
]


@pytest.mark.parametrize('name,params', problems)
def test_problems(name, params):
    X, F, CV = load("problems", name)
    F = at_least_2d_array(F, extend_as="col")

    problem = get_problem(name, *params)
    _F, _G = problem.evaluate(X, return_values_of=["F", "G"])

    np.testing.assert_allclose(_F, F, rtol=0, atol=1e-4)

    if problem.has_constraints():
        _CV = calc_cv(_G)
        np.testing.assert_allclose(_CV, CV, rtol=0, atol=1e-4)


def load(*args, attrs=["x", "f", "cv"]):
    name = args[-1]
    return tuple([load_to_test_resource(*args[:-1], f"{name}.{attr}", to="numpy") for attr in attrs])
