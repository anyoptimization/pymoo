import numpy as np
import pymoo.gradient.toolbox as anp


from pymoo.util.ref_dirs.construction import ConstructionBasedReferenceDirectionFactory
from pymoo.util.ref_dirs.misc import project_onto_sum_equals_zero_plane, project_onto_unit_simplex_recursive
from pymoo.util.ref_dirs.optimizer import Adam
from pymoo.util.ref_dirs.reduction import ReductionBasedReferenceDirectionFactory
from pymoo.util.reference_direction import ReferenceDirectionFactory, scale_reference_directions


class RieszEnergyReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self,
                 n_dim,
                 n_points,
                 ref_points=None,
                 return_as_tuple=False,
                 n_max_iter=1000,
                 n_until_optimizer_reset=30,
                 sampling="reduction",
                 norm_gradients=True,
                 verify_gradient=False,
                 freeze_edges=False,
                 precision=1e-5,
                 restarts=True,
                 X=None,
                 d=None,
                 callback=None,
                 **kwargs):

        super().__init__(n_dim, **kwargs)

        self.n_points = n_points
        self.n_max_iter = n_max_iter
        self.n_max_not_improved = n_until_optimizer_reset
        self.return_as_tuple = return_as_tuple
        self.sampling = sampling
        self.X = X
        self.ref_points = ref_points
        self.precision = precision
        self.verify_gradient = verify_gradient
        self.norm_gradients = norm_gradients
        self.freeze_edges = freeze_edges
        self.d = d
        self.callback = callback
        self.restarts = restarts

        # experiments have shown that dimensions squared is good value to choose here
        if self.d is None:
            self.d = n_dim * 2

    def _step(self, optimizer, X, freeze=None):
        free = np.logical_not(freeze)

        obj, grad, mutual_dist = calc_potential_energy_with_grad(X, self.d, return_mutual_dist=True)
        # obj, grad = value_and_grad(calc_potential_energy)(X, self.d)

        if self.verify_gradient:
            from autograd import value_and_grad
            obj, grad = calc_potential_energy_with_grad(X, self.d)
            _obj, _grad = value_and_grad(calc_potential_energy)(X, self.d)
            if np.abs(grad - _grad).mean() > 1e-5:
                print("GRADIENT IMPLEMENTATION IS INCORRECT!")

        # set the gradients for frozen points to zero - make them not to move
        if freeze is not None:
            grad[freeze] = 0

        # project the gradient to have a sum of zero - guarantees to stay on the simplex
        proj_grad = project_onto_sum_equals_zero_plane(grad)

        # normalize the gradients by the largest gradient norm
        if self.norm_gradients:
            norm = np.linalg.norm(proj_grad, axis=1)
            proj_grad = (proj_grad / max(norm.max(), 1e-24))

        # apply a step of gradient descent by subtracting the projected gradient with a learning rate
        X = optimizer.next(X, proj_grad)

        # project the out of bounds points back onto the unit simplex
        X[free] = project_onto_unit_simplex_recursive(X[free])

        # because of floating point issues make sure it is on the unit simplex
        X /= X.sum(axis=1)[:, None]

        return X, obj

    def _solve(self, X, F=None, freeze_edges=True):
        n_points = len(X)
        ret, obj = X, np.inf
        n_not_improved = 0

        # get the edge mask
        if freeze_edges:
            freeze = np.any(X < 1e-16, axis=1)
        else:
            freeze = np.full(len(X), False)

        # if additional points to be frozen are provided - add them to the X and mark them as frozen
        if F is not None:
            X = np.row_stack([X, F])
            freeze = np.concatenate([freeze, np.full(len(F), True)])

        # if all points are frozen - simply return it
        if np.all(freeze):
            return X

        # initialize the optimizer for the run
        self.optimizer = Adam(alpha=0.005)

        if self.callback is not None:
            self.callback(self, X)

        # for each iteration of gradient descent
        for i in range(self.n_max_iter):

            # execute one optimization step
            _X, _obj = self._step(self.optimizer, X, freeze=freeze)

            # if it is the current best solution -> store it
            if _obj < obj:
                ret, obj, n_not_improved = _X, _obj, 0
            else:
                n_not_improved += 1

            # evaluate how much the points have moved
            delta = np.sqrt((_X[:n_points] - X[:n_points]) ** 2).mean(axis=1).mean()

            if self.verbose:
                print(i, "objective", _obj, "delta", delta)

            # if there was only a little delta during the last iteration -> terminate
            if delta < self.precision or np.isnan(_obj):
                break

            # reset the optimizer if the objective value has not improved x iterations
            if self.restarts and n_not_improved > self.n_max_not_improved:
                self.optimizer = Adam(alpha=self.optimizer.alpha / 2)
                _X = ret
                n_not_improved = 0

            # otherwise use the new points for the next iteration
            X = _X

            if self.callback is not None:
                self.callback(self, X)

        return ret[:n_points]

    def _do(self):
        X = self.X

        # if no initial points are provided by the user
        if X is None:
            if self.sampling == "reduction":
                X = ReductionBasedReferenceDirectionFactory(self.n_dim,
                                                            self.n_points,
                                                            kmeans=True,
                                                            lexsort=False) \
                    .do()

            elif self.sampling == "construction":
                X = ConstructionBasedReferenceDirectionFactory(self.n_dim,
                                                               self.n_points) \
                    .do()
            else:
                raise Exception("Unknown sampling method. Either reduction or construction.")

        X = self._solve(X, freeze_edges=self.freeze_edges)

        if self.ref_points is not None:
            X, R = self.calc_ref_points(X, self.ref_points)

            if self.return_as_tuple:
                return X, R
            else:
                return np.row_stack([X, R])

        return X

    def calc_ref_points(self, X, ref_points):
        n_points = len(X)

        # the center needed for translations
        centroid = np.full((1, self.n_dim), 1 / self.n_dim)

        R = []

        # for each reference point provided by the user
        for entry in ref_points:
            ref_point, n_points_of_ref = entry.get("coordinates"), entry.get("n_points")
            scale, volume = entry.get("scale"), entry.get("volume")

            if scale is None:
                if volume is None:
                    raise Exception("Either define scale or volume!")
                else:
                    scale = volume ** (self.n_dim - 1)

            # retrieve all points to consider
            _X = np.row_stack([X] + R)

            # translate X to make the simplex to fill the unit
            v = centroid - ref_point
            X_t = _X + v
            X_t = scale_reference_directions(X_t, 1 / scale)

            # find the indices of points which are used as edges
            I = np.where(np.any(X_t < 1e-5, axis=1))[0]

            # create new points in the simplex where reference directions are supposed to be optimized
            _n_points = n_points_of_ref + (n_points - len(I))
            _R = ReductionBasedReferenceDirectionFactory(self.n_dim, n_points=_n_points, kmeans=True,
                                                         lexsort=False).do()

            # detect the edges and just optimize them and shrink later
            outer = np.any(_R == 0, axis=1)
            inner = ~outer

            # rescale the reference directions to be not too close to existing points
            _R = scale_reference_directions(_R, 0.9)

            # optimize just the edges
            _R[outer] = self._solve(_R[outer], F=np.row_stack([X_t[I], _R[inner]]), freeze_edges=False)

            # no set the reference point and freeze it
            # closest_to_centroid = cdist(centroid, _R).argmin()
            # outer[closest_to_centroid] = True
            # inner[closest_to_centroid] = False
            # _R[closest_to_centroid] = centroid

            # now when translated the reference point becomes the centroid - fix that point to be included
            _R[inner] = self._solve(_R[inner], F=np.row_stack([X_t[I], _R[outer]]), freeze_edges=False)

            # rescale and translate them back
            _R_t = scale_reference_directions(_R, scale)
            _R_t = _R_t - v

            # if any point is out of bounds - volume was too large
            if not np.all(_R_t >= 0):

                # get the corner points if not transformed
                V = (np.eye(self.n_dim) - centroid)

                # get the corner points of the ref dir simplex
                P = ref_point + scale * V
                E = centroid + scale * V

                # project because at least is out of points
                P_proj = project_onto_unit_simplex_recursive(np.copy(P))

                for i in range(len(P)):

                    if not np.all(P[i] == P_proj[i]):
                        _R_t = scale_reference_directions(_R, scale)
                        _R_t = _R_t - (E[i] - P_proj[i])

                        if np.all(_R_t >= 0):
                            break

            n_points += n_points_of_ref
            R.append(_R_t)

            # filter out points from to be removed from the original array
            X = X[[i for i in I if i < len(X)]]

        R = np.row_stack(R)

        return X, R


# ---------------------------------------------------------------------------------------------------------
# Energy Functions
# ---------------------------------------------------------------------------------------------------------


def squared_dist(A, B):
    return ((A[:, None] - B[None, :]) ** 2).sum(axis=2)


def calc_potential_energy(A, d):
    i, j = anp.triu_indices(len(A), 1)
    D = anp.sqrt(squared_dist(A, A)[i, j])
    energy = anp.log((1 / D ** d).mean())
    return energy


def calc_potential_energy_with_grad(x, d, return_mutual_dist=False):
    diff = (x[:, None] - x[None, :])
    # calculate the squared euclidean from each point to another
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # make sure the distance to itself does not count
    np.fill_diagonal(dist, np.inf)

    # epsilon which is the smallest distance possible to avoid an overflow during gradient calculation
    eps = 10 ** (-320 / (d + 2))
    b = dist < eps
    dist[b] = eps

    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = dist[np.triu_indices(len(x), 1)]

    # calculate the energy by summing up the squared distances
    energy = (1 / mutual_dist ** d).sum()
    log_energy = - np.log(len(mutual_dist)) + np.log(energy)

    # calculate the gradient
    grad = (-d * diff) / (dist ** (d + 2))[..., None]
    grad = np.sum(grad, axis=1)
    grad /= energy

    ret = [log_energy, grad]
    if return_mutual_dist:
        ret.append(mutual_dist)

    return tuple(ret)
