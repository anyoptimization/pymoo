import numpy as np

from pymoo.operators.mutation.inversion_mutation import inversion_mutation


def test_inversion_mutation():
    y = np.array([1, 2, 3, 4, 5])
    start = 1
    end = 3

    mut = inversion_mutation(y, seq=(start, end))
    np.testing.assert_allclose(mut, np.array([1, 4, 3, 2, 5]))

