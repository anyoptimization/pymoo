from os.path import join

import autograd.numpy as anp
import pytest

from pymoo.problems import get_problem
from tests.problems.test_correctness import load
from tests.util import path_to_test_resource

import numpy as np

problems = [
    ('DF1', [10, 5]), ('DF2', [10, 5]), ('DF3', [10, 5]), ('DF4', [10, 5]), ('DF5', [10, 5]),
    ('DF5', [10, 5]), ('DF6', [10, 5]), ('DF7', [10, 5]), ('DF8', [10, 5]), ('DF9', [10, 5]),
    ('DF10', [10, 5]), ('DF11', [10, 5]), ('DF12', [10, 5]), ('DF13', [10, 5]), ('DF14', [10, 5])
]


@pytest.mark.parametrize('name,params', problems)
def test_problems(name, params):
    nt, taut = params

    problem = get_problem(name, nt=nt, taut=taut)

    X, F, _ = load(name, suffix=["DF"])

    if F is None:
        print("Warning: No correctness check for %s" % name)
        return

    _F = problem.evaluate(X)

    np.testing.assert_allclose(_F, F)


@pytest.mark.parametrize('name,params', problems)
def test_pf(name, params):
    nt, taut = params
    problem = get_problem(name, nt=nt, taut=taut)

    path = path_to_test_resource("problems", "DF")

    pf = np.loadtxt(join(path, "%s.pf" % name))

    _pf = problem.pareto_front(n_pareto_points=len(pf))

    np.testing.assert_allclose(_pf, pf)
