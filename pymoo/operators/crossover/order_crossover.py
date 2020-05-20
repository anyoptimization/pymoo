from pymoo.model.crossover import Crossover
import numpy as np


class OrderCrossover(Crossover):
    """
    Order crossover for permutation encoding proposed by Davis.
    """
    def __init__(self, prob=1, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # X is a list of pairs of parents: X[:, 1, :] is the 1th group of parents
        n_pair = X.shape[1]
        Y = np.full(X.shape, 0, dtype=problem.type_var)
        for i in range(n_pair):
            x1, x2 = X[:, i, :]
            y1, y2 = OrderCrossover.cross(x1, x2)
            Y[:, i, :] = [y1, y2]
        return Y

    @staticmethod
    def cross(x1, x2):
        assert len(x1) == len(x2)
        start, end = np.sort(np.random.choice(len(x1), 2, replace=False))
        y1 = x1.copy()
        y2 = x2.copy()
        # build y1 and y2
        segment1 = set(y1[start:end])
        segment2 = set(y2[start:end])
        I = np.concatenate((np.arange(0, start), np.arange(end, len(x1))))

        # find elements in x2 that are not in segment1
        y1[I] = [y for y in x2 if y not in segment1]
        # find elements in x1 that are not in segment2
        y2[I] = [y for y in x1 if y not in segment2]

        return y1, y2