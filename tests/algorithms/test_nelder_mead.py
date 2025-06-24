import numpy as np
from scipy.optimize import minimize as scipy_minimize

from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.termination import get_termination


def test_no_bounds():
    problem = get_problem("sphere", n_var=2)
    problem.xl = None
    problem.xu = None
    method = NelderMead(x0=np.array([1, 1]), max_restarts=0)
    minimize(problem, method, verbose=False)
    assert True


def test_with_bounds_no_restart():
    problem = get_problem("sphere", n_var=2)
    method = NelderMead(x0=np.array([1, 1]), max_restarts=0)
    minimize(problem, method, verbose=False)


def test_with_bounds_no_initial_point():
    problem = get_problem("sphere", n_var=2)
    method = NelderMead(max_restarts=0)
    minimize(problem, method, verbose=False)


def test_with_bounds_with_restart():
    problem = get_problem("sphere", n_var=2)
    method = NelderMead(x0=np.array([1, 1]), max_restarts=2)
    minimize(problem, method, verbose=False)


def test_against_scipy():
    problem = get_problem("rosenbrock")
    problem.xl = None
    problem.xu = None
    x0 = np.array([0.5, 0.5])

    hist_scipy = []

    def fun(x):
        hist_scipy.append(x)
        return problem.evaluate(x)

    scipy_minimize(fun, x0, method='Nelder-Mead')
    hist_scipy = np.array(hist_scipy)

    hist = []

    def callback(x, _):
        if x.shape == 2:
            hist.extend(x)
        else:
            hist.append(x)

    problem.callback = callback
    minimize(problem, NelderMead(x0=x0), termination=("n_eval", len(hist_scipy)))
    hist = np.vstack(hist)[:len(hist_scipy)]

    np.testing.assert_allclose(hist, hist_scipy, rtol=1e-04)
