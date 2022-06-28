import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad

from pymoo.util.normalization import normalize
from pymoo.util.ref_dirs.energy import squared_dist
from pymoo.util.ref_dirs.optimizer import Adam
from pymoo.util.reference_direction import ReferenceDirectionFactory, scale_reference_directions


class LayerwiseRieszEnergyReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self,
                 n_dim,
                 partitions,
                 return_as_tuple=False,
                 n_max_iter=1000,
                 verbose=False,
                 X=None,
                 **kwargs):

        super().__init__(n_dim, **kwargs)
        self.scalings = None
        self.n_max_iter = n_max_iter
        self.verbose = verbose
        self.return_as_tuple = return_as_tuple
        self.X = X
        self.partitions = partitions

    def _step(self, optimizer, X, scalings):
        obj, grad = value_and_grad(calc_potential_energy)(scalings, X)
        scalings = optimizer.next(scalings, np.array(grad))
        scalings = normalize(scalings, xl=0, xu=scalings.max())
        return scalings, obj

    def _solve(self, X, scalings):

        # initialize the optimizer for the run
        optimizer = Adam()

        # for each iteration of gradient descent
        for i in range(self.n_max_iter):

            # execute one optimization step
            _scalings, _obj = self._step(optimizer, X, scalings)

            # evaluate how much the points have moved
            delta = np.abs(_scalings - scalings).sum()

            if self.verbose:
                print(i, "objective", _obj, "delta", delta)

            # if there was only a little delta during the last iteration -> terminate
            if delta < 1e-5:
                scalings = _scalings
                break

            # otherwise use the new points for the next iteration
            scalings = _scalings

        self.scalings = scalings
        return get_points(X, scalings)

    def do(self):

        X = []
        scalings = []

        for k, p in enumerate(self.partitions):

            if p > 1:
                val = np.linspace(0, 1, p + 1)[1:-1]
                _X = []
                for i in range(self.n_dim):
                    for j in range(i + 1, self.n_dim):
                        x = np.zeros((len(val), self.n_dim))
                        x[:, i] = val
                        x[:, j] = 1 - val
                        _X.append(x)

                X.append(np.row_stack(_X + [np.eye(self.n_dim)]))

            elif p == 1:
                X.append(np.eye(self.n_dim))
            else:
                X.append(np.full(self.n_dim, 1 / self.n_dim)[None, :])

            scalings.append(1 - k / len(self.partitions))

        scalings = np.array(scalings)
        X = self._solve(X, scalings)

        return X


# ---------------------------------------------------------------------------------------------------------
# Energy Functions
# ---------------------------------------------------------------------------------------------------------


def get_points(X, scalings):
    vals = []
    for i in range(len(X)):
        vals.append(scale_reference_directions(X[i], scalings[i]))
    X = anp.row_stack(vals)
    return X


def calc_potential_energy(scalings, X):
    X = get_points(X, scalings)

    i, j = anp.triu_indices(len(X), 1)
    D = squared_dist(X, X)[i, j]

    if np.any(D < 1e-12):
        return np.nan, np.nan

    return (1 / D).mean()