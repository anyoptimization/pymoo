import numpy as np
import pytest

from pymoo.util.normalization import ZeroToOneNormalization

n_var = 10


@pytest.fixture
def matrix_input():
    xl, xu = np.full(n_var, -5.0), np.full(n_var, 5.0)
    X = np.random.random((200, n_var)) * (xu - xl) + xl
    return X, xl, xu


@pytest.fixture
def vector_input():
    xl, xu = np.full(n_var, -5.0), np.full(n_var, 5.0)
    X = np.random.random(n_var) * (xu - xl) + xl
    return X, xl, xu


def test_zero_to_one(matrix_input):
    X, xl, xu = matrix_input

    norm = ZeroToOneNormalization(xl, xu)
    N = norm.forward(X)
    _X = norm.backward(N)
    np.testing.assert_allclose(X, _X)

    # now let us do just backward
    norm = ZeroToOneNormalization(xl, xu)
    _X = norm.backward(N)
    np.testing.assert_allclose(X, _X)


def test_zero_to_one_xl_and_xu_equal(matrix_input):
    X, xl, xu = matrix_input
    xl[0] = 15.0
    xu[0] = 15.0

    X[:, 0] = 15.0

    norm = ZeroToOneNormalization(xl, xu)
    N = norm.forward(X)
    assert np.all(N[:, 0] == 0.0)

    _X = norm.backward(N)
    np.testing.assert_allclose(X, _X)


def test_zero_to_one_xl_has_nan(matrix_input):
    X, xl, xu = matrix_input

    xl[0] = np.nan
    xu[0] = np.nan

    norm = ZeroToOneNormalization(xl, xu)
    N = norm.forward(X)
    assert np.all(N[:, 0] == X[:, 0])

    _X = norm.backward(N)
    np.testing.assert_allclose(X, _X)


def test_zero_to_one_only_one_dim(vector_input):
    X, xl, xu = vector_input

    norm = ZeroToOneNormalization(xl, xu)
    N = norm.forward(X)
    _X = norm.backward(N)
    np.testing.assert_allclose(X, _X)


def test_zero_to_one_xl_and_xu_are_none(vector_input):
    X, xl, xu = vector_input

    norm = ZeroToOneNormalization(None, None)
    N = norm.forward(X)
    _X = norm.backward(N)
    np.testing.assert_allclose(X, _X)


def test_none_as_input(vector_input):
    X, xl, xu = vector_input
    norm = ZeroToOneNormalization(xl, xu)
    N = norm.forward(None)
    assert N is None


def test_only_xl(vector_input):
    X, xl, _ = vector_input
    norm = ZeroToOneNormalization(xl, None)

    N = norm.forward(xl)
    assert np.all(N == 0.0)

    np.testing.assert_allclose(X, norm.backward(norm.forward(X)))


def test_only_xu(vector_input):
    X, _, xu = vector_input
    norm = ZeroToOneNormalization(None, xu)

    N = norm.forward(xu)
    assert np.all(N == 1.0)

    np.testing.assert_allclose(X, norm.backward(norm.forward(X)))



