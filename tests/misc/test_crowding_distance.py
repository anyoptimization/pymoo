import os

import numpy as np
import pytest

from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance, FunctionalDiversity
from pymoo.config import get_pymoo

crowding_func = FunctionalDiversity(calc_crowding_distance, filter_out_duplicates=True)

@pytest.mark.skip(reason="check if this is supposed to work or not at all")
def test_crowding_distance():
    D = np.loadtxt(os.path.join(get_pymoo(), "pytest", "resources", "test_crowding.dat"))
    F, cd = D[:, :-1], D[:, -1]
    assert np.all(np.abs(cd - crowding_func.do(F)) < 0.001)


def test_crowding_distance_one_duplicate():
    F = np.array([[1.0, 1.0], [1.0, 1.0], [0.5, 1.5], [0.0, 2.0]])
    cd = crowding_func.do(F)
    np.testing.assert_almost_equal(cd, np.array([np.inf, 0.0, 1.0, np.inf]))


def test_crowding_distance_two_duplicates():
    F = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.5, 1.5], [0.0, 2.0]])
    cd = crowding_func.do(F)
    np.testing.assert_almost_equal(cd, np.array([np.inf, 0.0, 0.0, 1.0, np.inf]))


def test_crowding_distance_norm_equals_zero():
    F = np.array([[1.0, 1.5, 0.5, 1.0], [1.0, 0.5, 1.5, 1.0], [1.0, 0.0, 2.0, 1.5]])
    cd = crowding_func.do(F)
    np.testing.assert_almost_equal(cd, np.array([np.inf, 0.75, np.inf]))

