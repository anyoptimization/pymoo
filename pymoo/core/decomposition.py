import numpy as np

from pymoo.util.misc import at_least_2d_array, to_1d_array_if_possible


class Decomposition:

    def __init__(self, eps=0.0, _type="auto", **kwargs) -> None:
        super().__init__()
        self.eps = eps
        self._type = _type
        self.ideal_point, self.utopian_point, self.nadir_point = None, None, None

    def __call__(self, *args, **kwargs):
        return self.do(*args, **kwargs)

    def do(self,
           F,
           weights,
           _type="auto",
           ideal_point=None,
           utopian_point=None,
           nadir_point=None,
           **kwargs):

        _F, _weights = to_1d_array_if_possible(F), to_1d_array_if_possible(weights)

        if _type == "auto":
            if _F.ndim == 1 and _weights.ndim > 1:
                _type = "one_to_many"
            elif _F.ndim > 1 and _weights.ndim == 1:
                _type = "many_to_one"
            elif _F.ndim == 2 and _weights.ndim == 2 and _F.shape[0] == _weights.shape[0]:
                _type = "one_to_one"
            else:
                _type = "many_to_many"

        # make both at least 2d arrays
        F, weights = at_least_2d_array(F), at_least_2d_array(weights)

        # get the number of points and weights
        n_points, n_weights = F.shape[0], weights.shape[0]

        self.ideal_point = ideal_point
        if self.ideal_point is None:
            self.ideal_point = np.zeros(F.shape[1])

        self.utopian_point = utopian_point
        if self.utopian_point is None:
            self.utopian_point = self.ideal_point - self.eps

        # set the nadir point by default to value or default
        self.nadir_point = nadir_point
        if self.nadir_point is None:
            self.nadir_point = self.utopian_point + np.ones(F.shape[1])

        if _type == "one_to_one":
            D = self._do(F, weights=weights, **kwargs).flatten()

        elif _type == "one_to_many":
            F = np.repeat(F, n_weights, axis=0)
            D = self._do(F, weights=weights, **kwargs).flatten()

        elif _type == "many_to_one":
            weights = np.repeat(weights, n_points, axis=0)
            D = self._do(F, weights=weights, **kwargs).flatten()

        elif _type == "many_to_many":
            F = np.repeat(F, n_weights, axis=0)
            weights = np.tile(weights, (n_points, 1))
            D = self._do(F, weights=weights, **kwargs).reshape(n_points, n_weights)

        else:
            raise Exception("Unknown type for decomposition: %s" % _type)

        return D
