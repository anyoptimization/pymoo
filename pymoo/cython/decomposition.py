import numpy as np

from pymoo.cython.function_loader import load_function


class Decomposition:

    def do(self, F, **kwargs):
        return self._do(F, **kwargs)


class PenaltyBasedBoundaryIntersection(Decomposition):

    def __init__(self, theta) -> None:
        super().__init__()
        self.theta = theta

    def _do(self, F, weights, ideal_point, **kwargs):

        n_points, n_weights = F.shape[0], weights.shape[0]

        if n_points == n_weights:
            pass
        elif n_points == 1 and n_weights > 1:
            F = np.repeat(F, n_weights, axis=0)
        elif n_points > 1 and n_weights == 1:
            weights = np.repeat(weights, n_points, axis=0)
        else:
            raise Exception("Either for each point a weight, one weight, or one objective value.")

        func = load_function("decomposition", "pbi")
        return func(F, weights, ideal_point, self.theta)


class WeightedSum(Decomposition):

    def _do(self, F, weights, **kwargs):
        return np.sum(F * weights, axis=1)


class Tchebicheff(Decomposition):

    def _do(self, F, weights, ideal_point, eps=1e-10, **kwargs):
        v = np.abs(F - ideal_point + eps) * weights
        return np.max(v, axis=1)


def pbi(F, weights, ideal_point, theta, eps=1e-10):
    utopian_point = ideal_point - eps
    norm = np.linalg.norm(weights, axis=1)
    d1 = np.sum((F - utopian_point) * weights, axis=1) / norm
    d2 = np.linalg.norm(F - utopian_point - (d1[:, None] * weights / norm[:, None]), axis=1)
    return d1 + theta * d2
