import numpy as np
from pymoo.util.remote import Remote

from pymoo.decomposition.perp_dist import PerpendicularDistance
from pymoo.decomposition.weighted_sum import WeightedSum


def test_one_to_one():
    F = np.random.random((2, 2))
    weights = np.array([[0.5, 0.5], [0.25, 0.25]])
    val = WeightedSum().do(F, weights=weights)

    assert val.ndim == 1
    assert val.shape[0] == 2


def test_one_to_many():
    F = np.random.random((1, 2))
    weights = np.array([[0.5, 0.5], [0.25, 0.25]])
    val = WeightedSum().do(F, weights=weights)

    assert val.ndim == 1
    assert val.shape[0] == 2


def test_many_to_one():
    F = np.random.random((10, 2))
    weights = np.array([[0.5, 0.5]])
    val = WeightedSum().do(F, weights=weights)

    assert val.ndim == 1
    assert val.shape[0] == 10


def test_many_to_many():
    F = np.random.random((10, 2))
    weights = np.array([[0.5, 0.5], [0.25, 0.25]])
    val = WeightedSum().do(F, weights=weights)

    assert val.shape[0] == 10
    assert val.shape[1] == 2


def test_perp_dist():
    np.random.seed(1)
    F = np.random.random((100, 3))
    weights = np.random.random((10, 3))

    correct = Remote.get_instance().load("tests", "perp_dist")

    D = PerpendicularDistance(_type="python").do(F, weights, _type="many_to_many")
    np.testing.assert_allclose(D, correct)

    D = PerpendicularDistance(_type="cython").do(F, weights, _type="many_to_many")
    np.testing.assert_allclose(D, correct)
