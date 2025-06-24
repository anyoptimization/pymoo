import numpy as np
import pytest

from pymoo.problems import get_problem
from tests.problems.test_correctness import load
from tests.test_util import load_to_test_resource

problems = [
    ('DF1', [10, 5]), ('DF2', [10, 5]), ('DF3', [10, 5]), ('DF4', [10, 5]), ('DF5', [10, 5]),
    ('DF6', [10, 5]), ('DF7', [10, 5]), ('DF8', [10, 5]), ('DF9', [10, 5]),
    ('DF10', [10, 5]), ('DF11', [10, 5]), ('DF12', [10, 5]), ('DF13', [10, 5]), ('DF14', [10, 5])
]


@pytest.mark.parametrize('name,params', problems)
def test_problems(name, params):

    nt, taut = params

    problem = get_problem(name, nt=nt, taut=taut)

    X, F, T = load("problems", "DF", name, attrs=["x", "f", "t"])

    _F = []
    i = 0
    for x_i in X:
        problem.tau = T[i]
        f_i = problem.evaluate(x_i)
        _F.append(f_i)
        i += 1
    _F = np.array(_F)

    np.testing.assert_allclose(_F, F)


@pytest.mark.parametrize('name,params', problems)
def test_pf(name, params):
    if name == 'DF11':
        pytest.skip("DF11 Pareto front calculation has numerical issues")

    nt, taut = 5, 1

    for tau in range(1, 11):
        problem = get_problem(name, nt=nt, taut=taut)
        file = name + '-' + str(tau) + '-' + str(taut) + '-' + str(nt) + ".pf"

        pf = load_to_test_resource("problems", "DF", "PF", file, to="numpy")

        if pf is not None and pf.shape[1] == 2:
            pf = pf[np.argsort(pf[:, 0])]
        problem.tau = tau
        if pf is not None and pf.shape[1] == 2:
            pf = pf[np.argsort(pf[:, 0])]
        _pf_t = problem.pareto_front(n_pareto_points=200)
        if _pf_t is not None and _pf_t.shape[1] == 2:
            _pf_t = _pf_t[np.argsort(_pf_t[:, 0])]

        np.testing.assert_allclose(pf, _pf_t, rtol=0, atol=1e-4)
