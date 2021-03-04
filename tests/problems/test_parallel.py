import multiprocessing

import numpy as np
import pytest

from pymoo.model.problem import ElementwiseProblem


class MyProblemElementwise(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x.sum()


@pytest.fixture
def data():
    np.random.seed(1)
    X = np.random.random((100, 2))
    F = X.sum(axis=1)[:, None]
    return X, F


def test_evaluation_in_threads(data):
    X, F = data
    _F = MyProblemElementwise(parallelization="threads").evaluate(X)
    np.testing.assert_allclose(_F, F)


def test_evaluation_in_threads_number(data):
    X, F = data
    _F = MyProblemElementwise(parallelization=("threads", 2)).evaluate(X)
    np.testing.assert_allclose(_F, F)


def test_evaluation_with_multiprocessing_process_pool_starmap(data):
    X, F = data
    with multiprocessing.Pool() as pool:
        _F = MyProblemElementwise(parallelization=("starmap", pool.starmap)).evaluate(X)
    np.testing.assert_allclose(_F, F)


def test_evaluation_with_multiprocessing_thread_pool_starmap(data):
    X, F = data
    with multiprocessing.pool.ThreadPool() as pool:
        _F = MyProblemElementwise(parallelization=("starmap", pool.starmap)).evaluate(X)
    np.testing.assert_allclose(_F, F)
