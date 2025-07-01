import numpy as np
import pytest

from pymoo.problems import get_problem
from pymoo.gradient.automatic import AutomaticDifferentiation
from pymoo.util.misc import at_least_2d_array, at_least_2d
from tests.problems.test_correctness import load

problems = [
    ('G1', []), ('G2', []), ('G3', []), ('G4', []), ('G5', []), ('G6', []), ('G7', []), ('G8', []),
    ('G9', []), ('G10', []), ('G11', []), ('G12', []), ('G13', []), ('G14', []), ('G15', []), ('G16', []),
    ('G17', []), ('G18', []), ('G19', []), ('G20', []), ('G21', []), ('G22', []), ('G23', []), ('G24', [])
]


@pytest.mark.parametrize('name,params', problems)
def test_problems(name, params):
    problem = get_problem(name, *params)

    xl, xu, ps, pf, X, F, G, H = load("problems", "G", name, attrs=["xl", "xu", "ps", "pf", "x", "f", "g", "h"])
    F, G, H = at_least_2d(F, G, H, extend_as="col")
    ps, pf = at_least_2d(ps, pf, extend_as="row")

    np.testing.assert_allclose(problem.xl, xl)
    np.testing.assert_allclose(problem.xu, xu)
    np.testing.assert_allclose(problem.pareto_set(), ps)
    np.testing.assert_allclose(problem.pareto_front(), pf)
    assert problem.n_var == X.shape[1]

    _F, _G, _H = problem.evaluate(X, return_values_of=["F", "G", "H"])

    np.testing.assert_allclose(_F, F)

    if problem.n_ieq_constr > 0:
        np.testing.assert_allclose(G, _G)

    if problem.n_eq_constr > 0:
        np.testing.assert_allclose(H, _H)

@pytest.mark.gradient
@pytest.mark.parametrize('name,params', problems)
def test_autodiff(name, params):
    problem = AutomaticDifferentiation(get_problem(name, *params))
    X = np.random.random((100, problem.n_var))
    problem.evaluate(X)
    assert True


