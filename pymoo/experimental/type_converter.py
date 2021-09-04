import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.misc import at_least_2d_array


class Converter:

    def __init__(self, attr="X") -> None:
        super().__init__()
        self.attr = attr
        self.obj = None

    def forward(self, obj, **kwargs):
        self.obj = obj
        return self._forward(obj, **kwargs)

    def backward(self, obj, **kwargs):
        return self._backward(obj, **kwargs)

    def _forward(self, obj, **kwargs):
        return obj

    def _backward(self, obj, **kwargs):
        return obj


class PopulationToNumpy(Converter):

    def _forward(self, pop, **kwargs):
        assert self.attr is not None and isinstance(pop, Population)
        return pop.get(self.attr)

    def _backward(self, X, inplace=False, **kwargs):
        assert isinstance(X, np.ndarray) and X.ndim == 2
        if inplace:
            self.obj.set(self.attr, X)
            return self.obj
        else:
            return Population.new(**{self.attr: X})


class IndividualToNumpy(Converter):

    def _forward(self, ind, **kwargs):
        assert self.attr is not None and isinstance(ind, Individual)
        return at_least_2d_array(ind.get(self.attr), extend_as="row")

    def _backward(self, X, inplace=False, **kwargs):
        assert isinstance(X, np.ndarray)

        # if it should be 2d we convert it to be a 1-d array
        if X.ndim == 2 and len(X) == 1:
            vals = X[0]

        assert vals.ndim == 1

        if inplace:
            self.obj.set(self.attr, vals)
            return self.obj
        else:
            return Individual(**{self.attr: vals})


class OneToTwoDimensionalNumpy(Converter):

    def _forward(self, x, **kwargs):
        assert isinstance(x, np.ndarray)
        if x.ndim == 1:
            return x[None, :]
        elif x.ndim == 2:
            return x
        else:
            raise Exception("Error while converting numpy array to 2d.")

    def _backward(self, x, inplace=False):
        assert isinstance(x, np.ndarray)

        # make sure it is actually 2d
        if x.ndim == 1:
            return x[None, :]
        elif x.ndim == 2:
            pass
        else:
            raise Exception("Error while converting numpy array to 2d.")

        if inplace:
            self.obj[:] = x
            return self.obj
        else:
            return x[0]


def get_converter(val):
    if isinstance(val, Population):
        return PopulationToNumpy()
    elif isinstance(val, Individual):
        return IndividualToNumpy()
    elif isinstance(val, np.ndarray):
        if val.ndim == 1:
            return OneToTwoDimensionalNumpy()
        elif val.ndim == 2:
            return Converter()
        else:
            raise Exception("No converter for numpy arrays with dimensions more than 2 exists.")


class DefaultConverter(Converter):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.converter = None

    def forward(self, obj, **kwargs):
        self.converter = get_converter(obj)
        return self.converter.forward(obj, **kwargs)

    def backward(self, obj, **kwargs):
        return self.converter.backward(obj, **kwargs)
