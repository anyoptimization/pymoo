import numpy as np
import pytest

from pymoo.constraints.from_bounds import ConstraintsFromBounds
from pymoo.problems import get_problem
from pymoo.indicators.kktpm import KKTPM
from pymoo.gradient.automatic import AutomaticDifferentiation
from tests.problems.test_correctness import load

SETUP = {
    "bnh": {'utopian_eps': 0.0, "ideal": np.array([-0.05, -0.05]), "rho": 0.0},
    "zdt1": {'utopian_eps': 1e-3},
    "zdt2": {'utopian_eps': 1e-4},
    "zdt3": {'utopian_eps': 1e-4, "ideal": np.array([0.0, -1.0])},
    # "osy": {'utopian_eps': 0.0, "ideal": np.array([-300, -0.05]), "rho": 0.0}
}


@pytest.mark.parametrize('str_problem,params', SETUP.items())
def test_kktpm_correctness(str_problem, params):
    problem = get_problem(str_problem)
    problem = ConstraintsFromBounds(AutomaticDifferentiation(problem))

    X, F, G, dF, dG, kktpm = load("kktpm", str_problem, attrs=["x", "f", "g", "df", "dg", "kktpm"])
    _F, _G, _dF, _dG = problem.evaluate(X, return_values_of=["F", "G", "dF", "dG"])

    dF = dF.reshape(_dF.shape)

    np.testing.assert_almost_equal(F, _F, decimal=5)
    np.testing.assert_almost_equal(dF, _dF, decimal=5)

    if problem.has_constraints():
        G = G[:, :problem.n_ieq_constr]
        dG = dG[:, :problem.n_ieq_constr * problem.n_var].reshape(_dG.shape)

        np.testing.assert_almost_equal(G, _G, decimal=5)
        np.testing.assert_almost_equal(dG, _dG, decimal=5)

    # indices = np.random.permutation(X.shape[0])[:100]
    n, _ = X.shape
    indices = np.arange(n)

    # calculate the KKTPM measure
    _kktpm = KKTPM().calc(X[indices], problem, **params)

    error = np.abs(_kktpm - kktpm)
    for i in range(len(error)):

        if error[i] > 0.0001:
            print("Error for ", str_problem)
            print("index: ", i)
            print("Error: ", error[i])
            print("X", ",".join(np.char.mod('%f', X[i])))
            print("Python: ", _kktpm[i])
            print("Correct: ", kktpm[i])

            # os._exit(1)

    # make sure the results are almost equal
    np.testing.assert_almost_equal(kktpm, _kktpm, decimal=4)
    print(str_problem, error.mean())
