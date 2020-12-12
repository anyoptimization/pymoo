import unittest

import numpy as np

from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.util.type_converter import IndividualToNumpy, PopulationToNumpy, OneToTwoDimensionalNumpy, get_converter, \
    DefaultConverter


class TypeConverterTest(unittest.TestCase):

    def test_numpy(self):
        conv = DefaultConverter()
        X = np.random.random(10)

        _X = conv.forward(X)
        assert _X.ndim == 2 and _X.shape == (1, 10)
        np.testing.assert_allclose(X[None, :], _X)

        Y = np.random.random((1, 10))
        res = conv.backward(Y)

        assert isinstance(res, np.ndarray) and not np.all(res == X)
        np.testing.assert_allclose(res, Y[0])

        res = conv.backward(Y, inplace=True)
        assert isinstance(res, np.ndarray) and np.all(res == X)
        np.testing.assert_allclose(res, Y[0])

    def test_population(self):
        conv = DefaultConverter()

        X = np.random.random((100, 10))
        pop = Population.new(X=X)

        _X = conv.forward(pop)
        assert _X.ndim == 2 and _X.shape == (100, 10)
        np.testing.assert_allclose(X, _X)

        Y = np.random.random((100, 10))
        res = conv.backward(Y)

        assert isinstance(res, Population) and not np.any(res == pop)
        np.testing.assert_allclose(res.get("X"), Y)

        res = conv.backward(Y, inplace=True)
        assert isinstance(res, Population) and np.all(res == pop)
        np.testing.assert_allclose(res.get("X"), Y)

    def test_individual(self):
        conv = DefaultConverter()
        X = np.random.random(10)

        ind = Individual(X=X)
        _X = conv.forward(ind)
        assert _X.ndim == 2 and _X.shape == (1, 10)
        np.testing.assert_allclose(X[None, :], _X)

        Y = np.random.random((1, 10))
        res = conv.backward(Y)

        assert isinstance(res, Individual) and res != ind
        np.testing.assert_allclose(res.get("X"), Y[0])

        res = conv.backward(Y, inplace=True)
        assert isinstance(res, Individual) and res == ind
        np.testing.assert_allclose(res.get("X"), Y[0])



if __name__ == '__main__':
    unittest.main()
