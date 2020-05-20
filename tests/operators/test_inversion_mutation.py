import unittest

import numpy as np

from pymoo.operators.crossover.order_crossover import ox
from pymoo.operators.mutation.inversion_mutation import inversion_mutation


class InversionMutationTest(unittest.TestCase):

    def test_example(self):
        y = np.array([1, 2, 3, 4, 5])
        start = 1
        end = 3

        mut = inversion_mutation(y, seq=(start, end))
        np.testing.assert_allclose(mut, np.array([1, 4, 3, 2, 5]))


if __name__ == '__main__':
    unittest.main()
