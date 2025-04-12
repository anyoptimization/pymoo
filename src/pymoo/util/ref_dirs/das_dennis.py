import numpy as np
from scipy import special


class DasDennis:

    def __init__(self, n_partitions, n_dim, scaling=None):
        super().__init__()
        self.n_partitions = n_partitions
        self.n_dim = n_dim
        self.scaling = scaling

        self.stack = []
        self.stack.append(([], self.n_partitions))

    def number_of_points(self):
        return int(special.binom(self.n_dim + self.n_partitions - 1, self.n_partitions))

    def next(self, n_points=None):
        ret = []
        self.traverse(lambda p: ret.append(p), n_points)
        return np.array(ret)

    def has_next(self):
        return len(self.stack) > 0

    def traverse(self, func, n_points=None):

        if self.n_partitions == 0:
            return np.full((1, self.n_dim), 1 / self.n_dim)

        counter = 0

        while (n_points is None or counter < n_points) and len(self.stack) > 0:

            point, beta = self.stack.pop()

            if len(point) + 1 == self.n_dim:
                point.append(beta / (1.0 * self.n_partitions))

                if self.scaling is not None:
                    point = [p * self.scaling + ((1 - self.scaling) / len(point)) for p in point]

                func(point)
                counter += 1
            else:
                for i in range(beta + 1):
                    _point = list(point)
                    _point.append(1.0 * i / (1.0 * self.n_partitions))
                    self.stack.append((_point, beta - i))

        return counter
