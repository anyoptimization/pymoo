import os

import numpy as np
import pytest

from pymoo.factory import get_problem
from pymoo.util.misc import at_least_2d_array
from tests.problems.test_correctness import load
from tests.util import path_to_test_resource

problems = [
    ('G01', []), ('G02', []), ('G03', []), ('G04', []), ('G05', []), ('G06', []), ('G07', []), ('G08', []),
    ('G09', []), ('G10', []), ('G11', []), ('G12', []), ('G13', []), ('G14', []), ('G15', []), ('G16', []),
    ('G17', []), ('G18', []), ('G19', []), ('G20', []), ('G21', []), ('G22', []), ('G23', []), ('G24', [])
]


@pytest.mark.parametrize('name,params', problems)
def test_problems(name, params):
    problem = get_problem(name, *params)

    path = path_to_test_resource("problems", "G")
    xl = np.loadtxt(os.path.join(path, "%s.xl" % name))
    np.testing.assert_allclose(problem.xl, xl)

    xu = np.loadtxt(os.path.join(path, "%s.xu" % name))
    np.testing.assert_allclose(problem.xu, xu)

    ps = problem.pareto_set()
    _ps = at_least_2d_array(np.loadtxt(os.path.join(path, "%s.ps" % name)), extend_as='r')
    np.testing.assert_allclose(_ps, ps)

    pf = problem.pareto_front()[0, 0]
    _pf = np.loadtxt(os.path.join(path, "%s.pf" % name)).flatten()[0]
    np.testing.assert_allclose(_pf, pf)

    X, F, _ = load(name, "G")

    assert problem.n_var == X.shape[1]

    _F, _G, _H = problem.evaluate(X, return_values_of=["F", "G", "H"])

    np.testing.assert_allclose(F, _F[:, 0])

    if _G is not None:
        G = at_least_2d_array(np.loadtxt(os.path.join(path, "%s.g" % name)), extend_as='c')
        assert problem.n_ieq_constr == G.shape[1]
        np.testing.assert_allclose(G, _G)

    if _H is not None:
        H = at_least_2d_array(np.loadtxt(os.path.join(path, "%s.h" % name)), extend_as='c')
        assert problem.n_eq_constr == H.shape[1]
        np.testing.assert_allclose(H, _H)
