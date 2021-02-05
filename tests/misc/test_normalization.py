import unittest

import numpy as np

from surrogate.util.norm.standardization import Standardization
from surrogate.util.norm.zero_to_one import ZeroToOneNormalization


def foward_and_backward(norm, y=None):
    if y is None:
        y = np.random.random((100, 2))
    n = norm.forward(y)
    y_prime = norm.backward(n)
    return y, y_prime


class TestNormalization(unittest.TestCase):

    def test_zero_to_one(self):

        y, y_prime = foward_and_backward(ZeroToOneNormalization())
        np.testing.assert_allclose(y, y_prime)

        _, y_prime = foward_and_backward(Standardization(), y=y)
        np.testing.assert_allclose(y, y_prime)


if __name__ == '__main__':
    unittest.main()
